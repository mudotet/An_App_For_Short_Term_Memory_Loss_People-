from __future__ import annotations

import os
from typing import Optional


class ConversationSummarizer:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = None
        self.init_error: Optional[str] = None

        if not self.api_key:
            self.init_error = "OPENAI_API_KEY is not set."
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        except Exception as exc:
            self.init_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    def summarize(self, transcript: str, person_name: Optional[str] = None) -> str:
        transcript = transcript.strip()
        if not transcript:
            return "Không có nội dung hội thoại đủ rõ để tóm tắt."

        if not self.client:
            return self._fallback_summary(
                transcript,
                "Chưa gọi được GPT vì thiếu hoặc lỗi cấu hình OPENAI_API_KEY.",
            )

        system_prompt = (
            "Bạn là trợ lý trí nhớ cho người có vấn đề về trí nhớ ngắn hạn. "
            "Hãy tóm tắt hội thoại thành 1-2 câu tiếng Việt, tập trung vào tên người "
            "nếu có, chủ đề chính, quyết định, lời hứa, thời gian, địa điểm hoặc thông tin cần nhớ. "
            "Không bịa thông tin nếu transcript không nói rõ."
        )
        user_prompt = (
            f"Tên người đang nói chuyện trong hệ thống: {person_name or 'chưa rõ'}\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Tóm tắt ngắn gọn để hiển thị trên kính thông minh:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=160,
            )
            summary = response.choices[0].message.content
            if summary:
                return " ".join(summary.strip().split())
        except Exception as exc:
            return self._fallback_summary(transcript, f"Không gọi được GPT: {exc}")

        return self._fallback_summary(transcript, "GPT không trả về nội dung tóm tắt.")

    @staticmethod
    def _fallback_summary(transcript: str, reason: str) -> str:
        excerpt = " ".join(transcript.split())[:240]
        suffix = "..." if len(transcript) > 240 else ""
        return f"{reason} Ghi chú tạm: {excerpt}{suffix}"
