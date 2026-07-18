from __future__ import annotations

import os
import math
import textwrap
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
) -> np.ndarray:
    return _draw_unicode_text(frame, labels)


def _draw_unicode_text(frame: np.ndarray, labels: Sequence[FaceLabel]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    now = time.monotonic()
    active_keys: set[tuple[str, object, str]] = set()
    occupied: list[tuple[float, float, int, int]] = []

    for label in labels:
        for kind, title, body in _bubble_specs(label):
            key = _label_key(label, kind)
            active_keys.add(key)
            content_key = (kind, title, body, label.is_new, frame.shape[1])
            state = _BUBBLE_STATES.get(key)

            if state is None or state.content_key != content_key:
                bubble, width, height = _build_bubble(
                    title,
                    body,
                    kind,
                    label.is_new,
                    frame.shape[1],
                    Image,
                    ImageDraw,
                    ImageFilter,
                    ImageFont,
                )
                if state is None:
                    state = _BubbleMotion(
                        label=label,
                        bubble=bubble,
                        content_key=content_key,
                        width=width,
                        height=height,
                        x=0.0,
                        y=0.0,
                        appeared_at=now,
                        updated_at=now,
                        last_seen=now,
                    )
                    _BUBBLE_STATES[key] = state
                else:
                    state.bubble = bubble
                    state.content_key = content_key
                    state.width = width
                    state.height = height

            state.label = label
            state.last_seen = now
            target_x, target_y = _bubble_target(
                label,
                state.width,
                state.height,
                frame.shape,
                kind,
            )
            target_x, target_y = _avoid_overlap(
                target_x,
                target_y,
                state.width,
                state.height,
                occupied,
                frame.shape[0],
            )
            occupied.append((target_x, target_y, state.width, state.height))

            if state.x == 0 and state.y == 0:
                state.x, state.y = target_x, target_y + 14
            elapsed = min(0.1, max(0.0, now - state.updated_at))
            blend = 1 - math.exp(-14 * elapsed)
            state.x += (target_x - state.x) * blend
            state.y += (target_y - state.y) * blend
            state.updated_at = now

    for key, state in list(_BUBBLE_STATES.items()):
        if key not in active_keys and now - state.last_seen > _EXIT_SECONDS:
            del _BUBBLE_STATES[key]

    if not _BUBBLE_STATES:
        _draw_idle_chip(image, now, Image, ImageDraw, ImageFont)
    else:
        for state in _BUBBLE_STATES.values():
            _paste_animated_bubble(image, state, now, Image, ImageDraw)

    output = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = output
    return frame


def _bubble_specs(label: FaceLabel) -> list[tuple[str, str, str]]:
    identity = label.title or ("Người lạ" if label.is_new else "Người quen")
    return [("identity", identity, label.subtitle)]


def _wrap_lines(text: str, width: int, max_lines: int) -> list[str]:
    lines = textwrap.wrap(" ".join(text.split()), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    return lines


@dataclass
class _BubbleMotion:
    label: FaceLabel
    bubble: object
    content_key: tuple[str, str, str, bool, int]
    width: int
    height: int
    x: float
    y: float
    appeared_at: float
    updated_at: float
    last_seen: float


_BUBBLE_MARGIN = 22
_EXIT_SECONDS = 0.28
_BUBBLE_STATES: dict[tuple[str, object, str], _BubbleMotion] = {}
_FONT_CACHE: dict[tuple[int, str], object] = {}


def _label_key(label: FaceLabel, kind: str) -> tuple[str, object, str]:
    identity = label.user_id if label.user_id is not None else label.title
    return ("user" if label.user_id is not None else "name", identity, kind)


def _bubble_target(
    label: FaceLabel,
    width: int,
    height: int,
    frame_shape: tuple[int, ...],
    kind: str,
) -> tuple[float, float]:
    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = label.location
    right_space = frame_width - right
    if kind == "identity" and top >= height - _BUBBLE_MARGIN + 8:
        x = (left + right - width) / 2
        y = top - (height - _BUBBLE_MARGIN) - 8
    elif right_space >= width:
        x = right + 10 - _BUBBLE_MARGIN
        y = top - 20
    elif left >= width:
        x = left - width - 10 + _BUBBLE_MARGIN
        y = top - 20
    else:
        x = (left + right - width) / 2
        y = (
            bottom + 10 - _BUBBLE_MARGIN
            if bottom + height - 2 * _BUBBLE_MARGIN + 10 <= frame_height
            else top - (height - _BUBBLE_MARGIN) - 10
        )
    return (
        float(max(4, min(x, frame_width - width - 4))),
        float(max(4, min(y, frame_height - height - 4))),
    )


def _avoid_overlap(
    x: float,
    y: float,
    width: int,
    height: int,
    occupied: Sequence[tuple[float, float, int, int]],
    frame_height: int,
) -> tuple[float, float]:
    for other_x, other_y, other_width, other_height in occupied:
        if not _rects_overlap(
            (x, y, width, height),
            (other_x, other_y, other_width, other_height),
        ):
            continue
        below = other_y + other_height + 10
        y = below if below + height <= frame_height - 4 else max(4, other_y - height - 10)
    return x, y


def _rects_overlap(
    first: tuple[float, float, int, int],
    second: tuple[float, float, int, int],
) -> bool:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2


def _build_bubble(
    title: str,
    body: str,
    kind: str,
    is_new: bool,
    frame_width: int,
    image_module: object,
    image_draw_module: object,
    image_filter_module: object,
    image_font_module: object,
) -> tuple[object, int, int]:
    title_font = _load_font(image_font_module, 19, "Semibold")
    body_font = _load_font(image_font_module, 14, "Medium")
    probe = image_draw_module.Draw(image_module.new("RGBA", (1, 1)))
    body_lines = _wrap_lines(body, width=42, max_lines=3) if body else []
    title_box = probe.textbbox((0, 0), title, font=title_font)
    body_boxes = [probe.textbbox((0, 0), line, font=body_font) for line in body_lines]
    title_indent = 18 if kind == "identity" else 0
    text_width = max(
        [title_box[2] - title_box[0] + title_indent]
        + [box[2] - box[0] for box in body_boxes]
    )
    min_width = 160 if kind == "identity" else 260
    card_width = min(max(min_width, text_width + 36), min(360, frame_width - 36))
    title_height = title_box[3] - title_box[1]
    body_heights = [box[3] - box[1] for box in body_boxes]
    card_height = 20 + title_height + 18
    if body_lines:
        card_height += 10 + sum(body_heights) + 5 * (len(body_lines) - 1)

    width = card_width + _BUBBLE_MARGIN * 2
    height = card_height + _BUBBLE_MARGIN * 2
    bubble = image_module.new("RGBA", (width, height))
    card_box = (
        _BUBBLE_MARGIN,
        _BUBBLE_MARGIN,
        _BUBBLE_MARGIN + card_width,
        _BUBBLE_MARGIN + card_height,
    )

    shadow_mask = image_module.new("L", (width, height))
    image_draw_module.Draw(shadow_mask).rounded_rectangle(card_box, radius=22, fill=125)
    shadow_mask = shadow_mask.filter(image_filter_module.GaussianBlur(7))
    shadow = image_module.new("RGBA", (width, height), (58, 31, 108, 0))
    shadow.putalpha(shadow_mask)
    bubble.alpha_composite(shadow)

    yy, xx = np.mgrid[0:card_height, 0:card_width]
    mix = 0.62 * xx / max(1, card_width - 1) + 0.38 * yy / max(1, card_height - 1)
    stops = np.array([0.0, 0.25, 0.5, 0.74, 1.0])
    palette = np.array(
        [
            [255, 134, 184],
            [190, 112, 255],
            [90, 150, 255],
            [86, 201, 235],
            [255, 172, 108],
        ]
    )
    colors = np.stack(
        [np.interp(mix, stops, palette[:, channel]) for channel in range(3)],
        axis=-1,
    )
    glow = np.exp(-(((xx / card_width) - 0.12) ** 2 + ((yy / card_height) - 0.1) ** 2) / 0.1)[..., None]
    colors = np.clip(colors * 0.88 + glow * 18, 0, 255)
    gradient = np.empty((card_height, card_width, 4), dtype=np.uint8)
    gradient[:, :, :3] = colors.astype(np.uint8)
    gradient[:, :, 3] = 232
    gradient_image = image_module.fromarray(gradient, "RGBA")
    card_mask = image_module.new("L", (card_width, card_height))
    image_draw_module.Draw(card_mask).rounded_rectangle(
        (0, 0, card_width - 1, card_height - 1),
        radius=20,
        fill=255,
    )
    bubble.paste(gradient_image, (_BUBBLE_MARGIN, _BUBBLE_MARGIN), card_mask)

    draw = image_draw_module.Draw(bubble)
    draw.rounded_rectangle(card_box, radius=22, outline=(255, 255, 255, 105), width=1)
    title_y = _BUBBLE_MARGIN + 16
    title_x = _BUBBLE_MARGIN + 16
    if kind == "identity":
        dot_color = (255, 241, 207, 255) if is_new else (231, 247, 255, 255)
        draw.ellipse((title_x, title_y + 5, title_x + 8, title_y + 13), fill=dot_color)
        title_x += 16
    draw.text((title_x + 1, title_y + 1), title, font=title_font, fill=(38, 20, 74, 115))
    draw.text(
        (title_x, title_y),
        title,
        font=title_font,
        fill=(255, 255, 255, 255),
    )
    text_y = title_y + title_height + 12
    for line, line_height in zip(body_lines, body_heights):
        draw.text(
            (_BUBBLE_MARGIN + 17, text_y + 1),
            line,
            font=body_font,
            fill=(38, 20, 74, 105),
        )
        draw.text(
            (_BUBBLE_MARGIN + 16, text_y),
            line,
            font=body_font,
            fill=(255, 255, 255, 250),
        )
        text_y += line_height + 5
    return bubble, width, height


def _paste_animated_bubble(
    image: object,
    state: _BubbleMotion,
    now: float,
    image_module: object,
    image_draw_module: object,
) -> None:
    enter = min(1.0, max(0.0, (now - state.appeared_at) / 0.42))
    eased = _ease_out_back(enter)
    opacity = min(1.0, enter * 1.8)
    stale_for = now - state.last_seen
    if stale_for > 0:
        exit_progress = min(1.0, stale_for / _EXIT_SECONDS)
        opacity *= 1 - exit_progress * exit_progress
        eased *= 1 - 0.04 * exit_progress

    scale = 0.92 + 0.08 * eased
    bubble = state.bubble
    if abs(scale - 1) > 0.002:
        bubble = bubble.resize(
            (max(1, int(state.width * scale)), max(1, int(state.height * scale))),
            image_module.Resampling.LANCZOS,
        )
    if opacity < 0.999:
        bubble = bubble.copy()
        bubble.putalpha(bubble.getchannel("A").point(lambda alpha: int(alpha * opacity)))

    x = int(state.x - (bubble.width - state.width) / 2)
    y = int(state.y - (bubble.height - state.height) / 2)
    tail_layer = image_module.new("RGBA", image.size)
    draw = image_draw_module.Draw(tail_layer)
    card_left = int(state.x + _BUBBLE_MARGIN)
    card_right = int(state.x + state.width - _BUBBLE_MARGIN)
    card_top = int(state.y + _BUBBLE_MARGIN)
    card_bottom = int(state.y + state.height - _BUBBLE_MARGIN)
    tail_y = int(card_top + min(48, (state.height - 2 * _BUBBLE_MARGIN) * 0.45))
    face_center = (state.label.location[1] + state.label.location[3]) / 2
    face_center_y = (state.label.location[0] + state.label.location[2]) / 2
    tail_alpha = int(230 * opacity)
    if face_center < card_left:
        tail = [(card_left + 2, tail_y - 9), (card_left + 2, tail_y + 9), (card_left - 14, tail_y)]
    elif face_center > card_right:
        tail = [(card_right - 2, tail_y - 9), (card_right - 2, tail_y + 9), (card_right + 14, tail_y)]
    elif face_center_y >= (card_top + card_bottom) / 2:
        tail_x = int(max(card_left + 24, min(face_center, card_right - 24)))
        tail = [(tail_x - 9, card_bottom - 2), (tail_x + 9, card_bottom - 2), (tail_x, card_bottom + 14)]
    else:
        tail_x = int(max(card_left + 24, min(face_center, card_right - 24)))
        tail = [(tail_x - 9, card_top + 2), (tail_x + 9, card_top + 2), (tail_x, card_top - 14)]
    draw.polygon(tail, fill=(148, 103, 230, tail_alpha))
    image.alpha_composite(tail_layer)
    image.alpha_composite(bubble, (x, y))


def _draw_idle_chip(
    image: object,
    now: float,
    image_module: object,
    image_draw_module: object,
    image_font_module: object,
) -> None:
    layer = image_module.new("RGBA", image.size)
    draw = image_draw_module.Draw(layer)
    font = _load_font(image_font_module, 15, "Medium")
    text = "Đang tìm khuôn mặt…"
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 52
    draw.rounded_rectangle(
        (18, 18, 18 + width, 58),
        radius=20,
        fill=(32, 24, 58, 175),
        outline=(255, 255, 255, 70),
    )
    pulse = int(175 + 80 * (0.5 + 0.5 * math.sin(now * 4)))
    draw.ellipse((32, 33, 42, 43), fill=(229, 146, 255, pulse))
    draw.text((50, 29), text, font=font, fill=(255, 255, 255, 255))
    image.alpha_composite(layer)


def _ease_out_back(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    c1 = 1.45
    c3 = c1 + 1
    return 1 + c3 * (progress - 1) ** 3 + c1 * (progress - 1) ** 2


def _load_font(image_font_module: object, size: int, weight: str = "Regular"):
    key = (size, weight)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Avenir Next.ttc",
        "C:/Windows/Fonts/segoeuib.ttf" if weight == "Semibold" else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            font = image_font_module.truetype(path, size)
            try:
                font.set_variation_by_name(weight)
            except (AttributeError, OSError, ValueError):
                pass
            _FONT_CACHE[key] = font
            return font
    _FONT_CACHE[key] = image_font_module.load_default()
    return _FONT_CACHE[key]


def _self_check() -> None:
    _BUBBLE_STATES.clear()
    frame = np.full((480, 640, 3), 24, dtype=np.uint8)
    original = frame.copy()
    label = FaceLabel(
        location=(170, 410, 330, 230),
        title="Minh Anh",
        subtitle="Lần trước đã hẹn cà phê vào chiều thứ Sáu.",
        user_id=1,
    )
    draw_labels(frame, [label])
    assert len(_BUBBLE_STATES) == 1
    for state in _BUBBLE_STATES.values():
        state.appeared_at -= 1
    draw_labels(frame, [label])
    assert not np.array_equal(frame, original)
    assert np.array_equal(frame[250, 230], original[250, 230])
    assert _bubble_specs(FaceLabel((0, 1, 1, 0), "", is_new=True)) == [
        ("identity", "Người lạ", "")
    ]
    assert _bubble_specs(FaceLabel((0, 1, 1, 0), "", subtitle="Đã gặp.")) == [
        ("identity", "Người quen", "Đã gặp.")
    ]
    assert _rects_overlap((0, 0, 20, 20), (10, 10, 20, 20))
    assert not _rects_overlap((0, 0, 10, 10), (10, 10, 10, 10))
    _BUBBLE_STATES.clear()
    print("Face bubble self-check: OK")


if __name__ == "__main__":
    _self_check()
