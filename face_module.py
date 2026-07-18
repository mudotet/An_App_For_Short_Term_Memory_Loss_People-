from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import face_recognition
import numpy as np


FaceBox = Tuple[int, int, int, int]  # top, right, bottom, left


@dataclass(frozen=True)
class FaceDetection:
    location: FaceBox
    embedding: list[float]


@dataclass(frozen=True)
class FaceLabel:
    location: FaceBox
    title: str
    subtitle: str = ""
    distance: Optional[float] = None
    is_new: bool = False
    user_id: Optional[int] = None


class FaceRecognizer:
    def __init__(self, frame_scale: float = 0.5, model: str = "hog", upsample: int = 1) -> None:
        if frame_scale <= 0 or frame_scale > 1:
            raise ValueError("frame_scale must be in the range (0, 1].")
        self.frame_scale = frame_scale
        self.model = model
        self.upsample = max(0, upsample)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame is None or frame.size == 0:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        working_frame = rgb_frame
        if self.frame_scale != 1:
            working_frame = cv2.resize(
                rgb_frame,
                (0, 0),
                fx=self.frame_scale,
                fy=self.frame_scale,
                interpolation=cv2.INTER_LINEAR,
            )

        small_locations = face_recognition.face_locations(
            working_frame,
            number_of_times_to_upsample=self.upsample,
            model=self.model,
        )
        encodings = face_recognition.face_encodings(working_frame, small_locations)

        detections: list[FaceDetection] = []
        for location, encoding in zip(small_locations, encodings):
            top, right, bottom, left = self._scale_location(location)
            detections.append(
                FaceDetection(
                    location=(top, right, bottom, left),
                    embedding=[float(value) for value in encoding],
                )
            )
        return detections

    def _scale_location(self, location: FaceBox) -> FaceBox:
        if self.frame_scale == 1:
            return location
        top, right, bottom, left = location
        factor = 1 / self.frame_scale
        return (
            int(top * factor),
            int(right * factor),
            int(bottom * factor),
            int(left * factor),
        )


def draw_labels(
    frame: np.ndarray,
    labels: Sequence[FaceLabel],
    unicode_text: bool = True,
) -> np.ndarray:
    if not labels:
        cv2.putText(
            frame,
            "No face detected",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    for label in labels:
        top, right, bottom, left = label.location
        color = (80, 220, 120) if not label.is_new else (80, 180, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    if unicode_text:
        try:
            return _draw_unicode_text(frame, labels)
        except Exception:
            pass
    return _draw_ascii_text(frame, labels)


def _draw_unicode_text(frame: np.ndarray, labels: Sequence[FaceLabel]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(ImageFont, 18)
    body_font = _load_font(ImageFont, 14)

    for label in labels:
        top, right, bottom, left = label.location
        lines = _label_lines(label)
        if not lines:
            continue

        measurements = [
            draw.textbbox((0, 0), text, font=title_font if idx == 0 else body_font)
            for idx, text in enumerate(lines)
        ]
        width = max(box[2] - box[0] for box in measurements) + 14
        line_heights = [box[3] - box[1] for box in measurements]
        height = sum(line_heights) + 8 + (len(lines) - 1) * 3

        x = max(4, min(left, frame.shape[1] - width - 4))
        y = top - height - 6
        if y < 4:
            y = min(bottom + 6, frame.shape[0] - height - 4)

        draw.rectangle((x, y, x + width, y + height), fill=(15, 20, 24, 215))
        text_y = y + 4
        for idx, text in enumerate(lines):
            font = title_font if idx == 0 else body_font
            color = (245, 250, 255, 255) if idx == 0 else (210, 230, 235, 255)
            draw.text((x + 7, text_y), text, font=font, fill=color)
            text_y += line_heights[idx] + 3

    output = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = output
    return frame


def _draw_ascii_text(frame: np.ndarray, labels: Sequence[FaceLabel]) -> np.ndarray:
    for label in labels:
        top, _, bottom, left = label.location
        lines = [_ascii(line) for line in _label_lines(label)]
        y = top - 8 if top > 36 else bottom + 22
        for line in lines:
            cv2.putText(
                frame,
                line,
                (max(4, left), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 20
    return frame


def _label_lines(label: FaceLabel) -> list[str]:
    distance_text = f" ({label.distance:.2f})" if label.distance is not None else ""
    title = f"{label.title}{distance_text}".strip()
    lines = [title]
    if label.subtitle:
        lines.extend(_wrap_lines(label.subtitle, width=42, max_lines=2))
    return lines


def _wrap_lines(text: str, width: int, max_lines: int) -> list[str]:
    lines = textwrap.wrap(" ".join(text.split()), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    return lines


_FONT_CACHE: dict[int, object] = {}


def _load_font(image_font_module: object, size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            _FONT_CACHE[size] = image_font_module.truetype(path, size)
            return _FONT_CACHE[size]
    _FONT_CACHE[size] = image_font_module.load_default()
    return _FONT_CACHE[size]


def _ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")
