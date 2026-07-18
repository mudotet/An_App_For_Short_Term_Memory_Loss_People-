# Trợ lý trí nhớ - demo hackathon

Ứng dụng demo mô phỏng kính thông minh cho người có vấn đề về trí nhớ ngắn hạn. Demo dùng webcam laptop để nhận diện nhiều khuôn mặt trong cùng khung hình, micro để ghi transcript tiếng Việt, PostgreSQL + pgvector để lưu embedding khuôn mặt, và OpenAI GPT để tóm tắt mỗi lần gặp.

## Demo

![Trợ lý trí nhớ nhận diện và ghi nhớ hội thoại](<docs/assets/Screen Recording 2026-07-18 at 14.11.30.gif>)

## Tính năng chính

- Phát hiện tất cả khuôn mặt trong frame webcam bằng `face_recognition`.
- Trích xuất embedding 128 chiều và so khớp trong PostgreSQL bằng pgvector.
- Người đã gặp: overlay tên + tóm tắt lần gặp gần nhất, đồng thời đọc lời nhắc bằng TTS.
- Người mới: tạo hồ sơ khuôn mặt mới, cho phép nhập tên tay để demo.
- Ghi âm theo từng đoạn ngắn, chuyển giọng nói tiếng Việt sang văn bản bằng `faster-whisper`.
- Cập nhật transcript rõ ràng lên bong bóng theo từng đoạn gần thời gian thực; âm thanh thiếu tin cậy sẽ không hiển thị.
- Khi người rời khung hình quá ngưỡng cấu hình hoặc bấm kết thúc, gọi GPT tóm tắt transcript và lưu thêm vào lịch sử hội thoại.

## Cấu trúc

```text
.
├── main.py              # Vòng lặp webcam, điều phối face/audio/db/llm
├── face_module.py       # Nhận diện khuôn mặt + overlay
├── audio_module.py      # faster-whisper STT + pyttsx3 TTS
├── db_module.py         # Kết nối PostgreSQL, query pgvector
├── llm_module.py        # Gọi OpenAI GPT để tóm tắt hội thoại
├── docker-compose.yml   # PostgreSQL kèm extension pgvector
├── db/init.sql          # Schema users/conversations
├── docs/assets          # GIF demo
├── requirements.txt
└── .env.example
```

## Chạy demo

1. Tạo file môi trường:

```bash
cp .env.example .env
```

Điền `OPENAI_API_KEY` trong `.env`. Nếu chưa có key, demo vẫn lưu transcript nhưng không hiển thị summary chưa được xác minh.

2. Khởi động database:

```bash
docker compose up -d
```

3. Tạo môi trường Python và cài thư viện:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Trên Windows, `face_recognition`/`dlib` có thể cần Visual Studio Build Tools hoặc cài qua Conda nếu pip build lỗi.

4. Chạy ứng dụng:

```bash
python main.py
```

Tùy chọn hữu ích:

```bash
python main.py --webcam-index 1
python main.py --no-audio
python main.py --no-tts
python main.py --auto-name-new-people
python main.py --fast --auto-name-new-people
python main.py --fast --accurate-detect --auto-name-new-people
```

Nếu camera bị giật trong lúc trình diễn, dùng `--fast`. Chế độ này giữ webcam ở khoảng 24 FPS và chạy nhận diện khuôn mặt ở luồng nền. Nếu khuôn mặt vẫn khó bắt, thêm `--accurate-detect`.

## Điều khiển khi demo

- `q`: thoát và lưu các phiên đang mở.
- `e`: kết thúc phiên hiện tại và lưu tóm tắt ngay.
- `m`: nhập ghi chú thủ công, hữu ích khi micro/STT gặp lỗi lúc trình diễn.

## Ghi chú kỹ thuật

- Ngưỡng nhận diện mặc định là `FACE_DISTANCE_THRESHOLD=0.62`, dùng khoảng cách Euclid/L2 của pgvector (`<->`).
- Với nhiều người trong cùng khung hình, mỗi người có một phiên hội thoại riêng. Vì laptop chỉ có một micro và demo chưa có speaker diarization, cùng một đoạn transcript sẽ được gắn cho các khuôn mặt đang xuất hiện trong khoảng thời gian đó.
- Schema lưu lịch sử không ghi đè: mỗi lần kết thúc phiên sẽ thêm một dòng mới vào bảng `conversations`.
- Nếu Docker volume đã được tạo trước khi có `db/init.sql`, hãy tạo lại database volume để schema được chạy lại.
