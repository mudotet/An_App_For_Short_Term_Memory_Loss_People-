from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional, Union

import cv2
from dotenv import load_dotenv

from audio_module import AudioTranscriber, ReminderSpeaker, TranscriptChunk
from db_module import MemoryDatabase
from face_module import FaceDetection, FaceLabel, FaceRecognizer, draw_labels
from llm_module import ConversationSummarizer


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


@dataclass
class AppConfig:
    webcam_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 24
    face_distance_threshold: float = 0.5
    face_absence_timeout_seconds: float = 8.0
    face_frame_scale: float = 0.5
    face_model: str = "hog"
    face_upsample: int = 1
    process_every_n_frames: int = 3
    face_process_interval_seconds: float = 0.25
    async_face_detection: bool = True
    unicode_overlay: bool = True
    show_fps: bool = True
    whisper_model: str = "small"
    whisper_language: str = "vi"
    audio_sample_rate: int = 16000
    audio_chunk_seconds: int = 5
    audio_enabled: bool = True
    tts_enabled: bool = True
    tts_cooldown_seconds: int = 90
    auto_name_new_people: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            webcam_index=int(os.getenv("WEBCAM_INDEX", "0")),
            camera_width=int(os.getenv("CAMERA_WIDTH", "640")),
            camera_height=int(os.getenv("CAMERA_HEIGHT", "480")),
            camera_fps=int(os.getenv("CAMERA_FPS", "24")),
            face_distance_threshold=float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.5")),
            face_absence_timeout_seconds=float(os.getenv("FACE_ABSENCE_TIMEOUT_SECONDS", "8")),
            face_frame_scale=float(os.getenv("FACE_FRAME_SCALE", "0.5")),
            face_model=os.getenv("FACE_MODEL", "hog"),
            face_upsample=max(0, int(os.getenv("FACE_UPSAMPLE", "1"))),
            process_every_n_frames=max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))),
            face_process_interval_seconds=max(
                0.0,
                float(os.getenv("FACE_PROCESS_INTERVAL_SECONDS", "0.25")),
            ),
            async_face_detection=_env_bool("ASYNC_FACE_DETECTION", True),
            unicode_overlay=_env_bool("UNICODE_OVERLAY", True),
            show_fps=_env_bool("SHOW_FPS", True),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "vi"),
            audio_sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            audio_chunk_seconds=max(1, int(os.getenv("AUDIO_CHUNK_SECONDS", "5"))),
            audio_enabled=_env_bool("AUDIO_ENABLED", True),
            tts_enabled=_env_bool("TTS_ENABLED", True),
            tts_cooldown_seconds=int(os.getenv("TTS_COOLDOWN_SECONDS", "90")),
            auto_name_new_people=_env_bool("AUTO_NAME_NEW_PEOPLE", False),
        )


@dataclass
class ConversationSession:
    user_id: int
    name: str
    latest_summary: str = ""
    started_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    transcript_parts: list[str] = field(default_factory=list)

    def add_transcript(self, chunk: Union[TranscriptChunk, str]) -> None:
        if isinstance(chunk, TranscriptChunk):
            timestamp = chunk.ended_at.strftime("%H:%M:%S")
            text = chunk.text
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            text = chunk
        text = " ".join(text.strip().split())
        if text:
            self.transcript_parts.append(f"[{timestamp}] {text}")

    def transcript(self) -> str:
        return "\n".join(self.transcript_parts).strip()


class MemoryAssistantApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.db = MemoryDatabase.from_env().connect()
        self.face_recognizer = FaceRecognizer(
            frame_scale=config.face_frame_scale,
            model=config.face_model,
            upsample=config.face_upsample,
        )
        self.audio = AudioTranscriber(
            model_size=config.whisper_model,
            language=config.whisper_language,
            sample_rate=config.audio_sample_rate,
            chunk_seconds=config.audio_chunk_seconds,
            enabled=config.audio_enabled,
        )
        self.speaker = ReminderSpeaker(
            enabled=config.tts_enabled,
            cooldown_seconds=config.tts_cooldown_seconds,
        )
        self.summarizer = ConversationSummarizer()
        self.sessions: dict[int, ConversationSession] = {}
        self.visible_user_ids: set[int] = set()
        self.latest_labels: list[FaceLabel] = []
        self.frame_number = 0
        self.last_face_process_at = 0.0
        self.face_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-worker")
        self.pending_face_future: Optional[Future[list[FaceLabel]]] = None
        self.state_lock = threading.RLock()
        self.display_fps = 0.0
        self._fps_window_started_at = time.monotonic()
        self._fps_window_frames = 0

    def run(self) -> int:
        self._print_startup_notes()
        self.audio.start()
        self.speaker.start()

        camera = cv2.VideoCapture(self.config.webcam_index)
        if not camera.isOpened():
            print(f"Không mở được webcam index {self.config.webcam_index}.")
            self._shutdown_workers()
            return 1
        self._configure_camera(camera)

        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Không đọc được frame từ webcam.")
                    break

                now = time.monotonic()
                self._consume_face_result()
                self._maybe_process_faces(frame, now)

                self._attach_audio_to_visible_people()
                self._finalize_absent_sessions()
                self._print_worker_errors()

                draw_labels(frame, self.latest_labels, unicode_text=self.config.unicode_overlay)
                self._draw_fps(frame)
                cv2.imshow("Memory Assistant Demo", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("e"):
                    self._finalize_selected_sessions("manual end")
                if key == ord("m"):
                    self._add_manual_note()

                self.frame_number += 1
                self._update_fps()
        finally:
            camera.release()
            cv2.destroyAllWindows()
            self._finalize_selected_sessions("shutdown", user_ids=list(self.sessions))
            self._shutdown_workers()
            self.face_executor.shutdown(wait=True, cancel_futures=True)
            self.db.close()

        return 0

    def _maybe_process_faces(self, frame, now: float) -> None:
        should_process = (
            self.frame_number % self.config.process_every_n_frames == 0
            and now - self.last_face_process_at >= self.config.face_process_interval_seconds
        )
        if not should_process:
            return
        if self.config.async_face_detection:
            if self.pending_face_future and not self.pending_face_future.done():
                return
            self.pending_face_future = self.face_executor.submit(self._recognize_frame, frame.copy())
            self.last_face_process_at = now
            return
        self.latest_labels = self._recognize_frame(frame)
        self.last_face_process_at = now

    def _consume_face_result(self) -> None:
        if not self.pending_face_future or not self.pending_face_future.done():
            return
        try:
            self.latest_labels = self.pending_face_future.result()
        except Exception as exc:
            print(f"Lỗi nhận diện khuôn mặt: {exc}")
        finally:
            self.pending_face_future = None

    def _configure_camera(self, camera) -> None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _update_fps(self) -> None:
        self._fps_window_frames += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_started_at
        if elapsed >= 1:
            self.display_fps = self._fps_window_frames / elapsed
            self._fps_window_frames = 0
            self._fps_window_started_at = now

    def _draw_fps(self, frame) -> None:
        if not self.config.show_fps:
            return
        cv2.putText(
            frame,
            f"FPS {self.display_fps:4.1f}",
            (24, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 255, 180),
            2,
            cv2.LINE_AA,
        )

    def _recognize_frame(self, frame) -> list[FaceLabel]:
        try:
            detections = self.face_recognizer.detect(frame)
        except Exception as exc:
            print(f"Lỗi nhận diện khuôn mặt: {exc}")
            with self.state_lock:
                self.visible_user_ids = set()
            return []

        if not detections:
            with self.state_lock:
                self.visible_user_ids = set()
            return []

        labels: list[FaceLabel] = []
        visible_user_ids: set[int] = set()
        now = time.monotonic()

        with self.state_lock:
            for detection in detections:
                try:
                    label = self._resolve_detection(detection, now)
                except Exception as exc:
                    print(f"Lỗi xử lý khuôn mặt: {exc}")
                    continue
                labels.append(label)
                session = self._session_for_label(label)
                if session:
                    visible_user_ids.add(session.user_id)

            self.visible_user_ids = visible_user_ids
        return labels

    def _resolve_detection(self, detection: FaceDetection, now: float) -> FaceLabel:
        match = self.db.find_matching_user(
            detection.embedding,
            threshold=self.config.face_distance_threshold,
        )

        if match:
            latest = self.db.get_latest_conversation(match.id)
            latest_summary = latest["summary"] if latest else "Đã gặp trước đó, chưa có tóm tắt hội thoại."
            is_new_session = match.id not in self.sessions
            session = self.sessions.get(match.id)
            if not session:
                session = ConversationSession(
                    user_id=match.id,
                    name=match.name,
                    latest_summary=latest_summary,
                    last_seen=now,
                )
                self.sessions[match.id] = session
            session.name = match.name
            session.latest_summary = latest_summary
            session.last_seen = now

            if is_new_session and latest:
                self.speaker.speak_once(
                    key=f"user-{match.id}",
                    message=f"Đây là {match.name}. Lần trước: {latest_summary}",
                )

            return FaceLabel(
                location=detection.location,
                title=match.name,
                subtitle=latest_summary,
                distance=match.distance,
                user_id=match.id,
            )

        user_id, name = self._register_new_person(detection.embedding)
        self.sessions[user_id] = ConversationSession(
            user_id=user_id,
            name=name,
            latest_summary="Người mới - đang ghi nhớ cuộc trò chuyện này.",
            last_seen=now,
        )
        return FaceLabel(
            location=detection.location,
            title=name,
            subtitle="Người mới - đang ghi nhớ cuộc trò chuyện này.",
            is_new=True,
            user_id=user_id,
        )

    def _register_new_person(self, embedding: list[float]) -> tuple[int, str]:
        default_name = f"Nguoi moi {datetime.now().strftime('%H%M%S')}"
        self.speaker.speak_once(
            key=f"new-person-{time.monotonic()}",
            message="Tôi đang gặp một người mới. Vui lòng nhập tên trên máy tính.",
        )
        print("\nPhát hiện người mới.")
        if self.config.auto_name_new_people or not sys.stdin.isatty():
            typed_name = ""
        else:
            try:
                typed_name = input(f"Nhập tên người này để demo [{default_name}]: ").strip()
            except EOFError:
                typed_name = ""
        name = typed_name or default_name
        user_id = self.db.create_user(name, embedding)
        print(f"Đã lưu khuôn mặt mới: {name} (user_id={user_id}).")
        return user_id, name

    def _session_for_label(self, label: FaceLabel) -> Optional[ConversationSession]:
        if label.user_id is None:
            return None
        return self.sessions.get(label.user_id)

    def _attach_audio_to_visible_people(self) -> None:
        chunks = self.audio.pop_transcripts()
        if not chunks:
            return

        with self.state_lock:
            targets = [self.sessions[user_id] for user_id in self.visible_user_ids if user_id in self.sessions]
        if not targets:
            print("Có transcript mới nhưng không có khuôn mặt nào trong khung hình, nên chưa gán vào hồ sơ.")
            return

        target_names = ", ".join(session.name for session in targets)
        for chunk in chunks:
            for session in targets:
                session.add_transcript(chunk)
            print(f"STT -> {target_names}: {chunk.text}")

    def _finalize_absent_sessions(self) -> None:
        now = time.monotonic()
        with self.state_lock:
            expired_user_ids = [
                user_id
                for user_id, session in self.sessions.items()
                if now - session.last_seen > self.config.face_absence_timeout_seconds
            ]
        if expired_user_ids:
            self._finalize_selected_sessions("face absent", user_ids=expired_user_ids)

    def _finalize_selected_sessions(
        self,
        reason: str,
        user_ids: Optional[Iterable[int]] = None,
    ) -> None:
        with self.state_lock:
            selected_ids = list(user_ids if user_ids is not None else (self.visible_user_ids or self.sessions.keys()))
            selected_sessions = [
                (user_id, self.sessions.pop(user_id))
                for user_id in selected_ids
                if user_id in self.sessions
            ]
            self.visible_user_ids.difference_update(selected_ids)

        for user_id, session in selected_sessions:
            transcript = session.transcript()
            try:
                if transcript:
                    summary = self.summarizer.summarize(transcript, person_name=session.name)
                    conversation_id = self.db.add_conversation(session.user_id, transcript, summary)
                    print(
                        f"Đã lưu hội thoại #{conversation_id} cho {session.name} "
                        f"({reason}): {summary}"
                    )
                else:
                    print(f"Kết thúc phiên của {session.name} ({reason}) nhưng chưa có transcript.")
                self.db.update_last_seen(session.user_id)
            except Exception as exc:
                print(f"Không lưu được phiên của {session.name}: {exc}")
                with self.state_lock:
                    self.sessions[user_id] = session

    def _add_manual_note(self) -> None:
        with self.state_lock:
            targets = [self.sessions[user_id] for user_id in self.visible_user_ids if user_id in self.sessions]
        if not targets:
            print("Chưa có người nào đang hiện trong khung hình để gắn ghi chú.")
            return
        try:
            note = input("Nhập ghi chú thủ công cho người đang thấy trong webcam: ").strip()
        except EOFError:
            note = ""
        if not note:
            return
        for session in targets:
            session.add_transcript(f"[manual] {note}")
        print(f"Đã gắn ghi chú thủ công cho: {', '.join(session.name for session in targets)}")

    def _print_worker_errors(self) -> None:
        for error in self.audio.pop_errors():
            print(error)
        for error in self.speaker.pop_errors():
            print(error)

    def _print_startup_notes(self) -> None:
        if not self.summarizer.is_ready:
            print(
                "Cảnh báo: chưa sẵn sàng gọi GPT. Ứng dụng vẫn lưu transcript "
                "và dùng tóm tắt tạm nếu thiếu OPENAI_API_KEY."
            )
        print("Demo đã sẵn sàng. Phím: q=thoát, e=kết thúc phiên hiện tại, m=thêm ghi chú tay.")

    def _shutdown_workers(self) -> None:
        self.audio.stop()
        self.speaker.stop()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Assistant webcam demo")
    parser.add_argument("--webcam-index", type=int, help="Override WEBCAM_INDEX.")
    parser.add_argument("--no-audio", action="store_true", help="Disable faster-whisper microphone STT.")
    parser.add_argument("--no-tts", action="store_true", help="Disable pyttsx3 reminders.")
    parser.add_argument("--fast", action="store_true", help="Prefer smoother webcam FPS for hackathon demos.")
    parser.add_argument("--sync-detect", action="store_true", help="Run face detection on the camera loop.")
    parser.add_argument("--accurate-detect", action="store_true", help="Prefer face detection sensitivity over FPS.")
    parser.add_argument(
        "--auto-name-new-people",
        action="store_true",
        help="Automatically name new faces instead of waiting for terminal input.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_console_encoding()
    load_dotenv()
    args = parse_args()
    config = AppConfig.from_env()
    if args.webcam_index is not None:
        config.webcam_index = args.webcam_index
    if args.no_audio:
        config.audio_enabled = False
    if args.no_tts:
        config.tts_enabled = False
    if args.auto_name_new_people:
        config.auto_name_new_people = True
    if args.fast:
        config.camera_width = 640
        config.camera_height = 480
        config.camera_fps = 24
        config.face_frame_scale = max(config.face_frame_scale, 0.5)
        config.face_upsample = max(config.face_upsample, 1)
        config.process_every_n_frames = max(config.process_every_n_frames, 4)
        config.face_process_interval_seconds = max(config.face_process_interval_seconds, 0.25)
        config.unicode_overlay = False
    if args.accurate_detect:
        config.face_frame_scale = max(config.face_frame_scale, 0.7)
        config.face_upsample = max(config.face_upsample, 2)
        config.process_every_n_frames = max(config.process_every_n_frames, 5)
        config.face_process_interval_seconds = max(config.face_process_interval_seconds, 0.35)
    if args.sync_detect:
        config.async_face_detection = False

    try:
        app = MemoryAssistantApp(config)
    except Exception as exc:
        print(f"Không khởi động được ứng dụng: {exc}")
        return 1

    return app.run()


if __name__ == "__main__":
    sys.exit(main())
