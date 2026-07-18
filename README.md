# MemoryLens AI — AI memory assistant for short-term memory support

MemoryLens AI logo

# MemoryLens AI

AI-assisted smart-glasses simulation — starting with face recall and conversation memory

Recognize people. Recall the last conversation. Save new memories after every meeting.

## Open Source Project

Topic: Assistive AI · Memory support · Human-centered smart devices

Problem: Short-term memory support for people who may forget recent encounters, names, and conversation context

Open-source prototype · Python · OpenCV · face_recognition · faster-whisper · PostgreSQL · pgvector · OpenAI API · Docker Compose

Project summary

## ✨ Product Tour

### Webcam smart-glasses simulation

MemoryLens AI uses a laptop webcam to simulate smart glasses that recognize people in real time and display memory prompts directly on the video frame.

### First-time encounter

When the system sees a new face, it creates a new profile, tracks the active conversation, transcribes speech, summarizes the encounter, and stores the memory with the detected face embedding.

### Returning encounter

When the same person appears again, the system retrieves their profile and shows the most recent conversation summary, for example:

> Đây là Minh — lần trước hai người bàn về việc đổi lịch họp.

## 💡 Problem Statement

People with short-term memory challenges, including early-stage Alzheimer’s or similar cognitive difficulties, can face three everyday obstacles:

- Forgetting recent encounters: they may not remember whether they have met someone before.
- Losing conversation context: names, topics, decisions, and promises can disappear quickly after a conversation ends.
- Social pressure: forgetting a person or a previous discussion can create stress, confusion, and avoidable embarrassment.

This prototype focuses on a practical assistive scenario: helping a user recognize people and remember the latest important conversation context through a webcam-based demo.

## ✨ Solution

MemoryLens AI turns a webcam and microphone into a demo version of memory-support smart glasses:

- Face recognition — detects one or more faces in a webcam frame and extracts 128-dimensional embeddings using `face_recognition`.
- Memory lookup — searches PostgreSQL with pgvector nearest-neighbor matching to find known faces.
- Real-time reminder — overlays the person’s name and latest conversation summary on the webcam feed.
- New-person capture — creates a new profile when no matching face is found.
- Vietnamese speech-to-text — records microphone audio and transcribes speech locally with `faster-whisper`.
- Conversation summary — sends the transcript to OpenAI GPT and saves a 1-2 sentence Vietnamese memory summary.
- History preservation — every encounter is stored as a new conversation record instead of overwriting the previous one.
- Voice reminder — optional TTS uses `pyttsx3` to speak short prompts.

Scope: this prototype is an assistive demo for memory support. It is not a medical device, diagnostic tool, surveillance system, or replacement for caregiver judgment.

## 🏗️ Architecture

MemoryLens AI system architecture covering webcam capture, face detection, face embedding, pgvector search, microphone recording, local Vietnamese STT, GPT summarization, TTS reminders, and OpenCV overlay.

```text
Laptop Webcam
    ↓
OpenCV Frame Loop
    ↓
face_recognition / dlib
    ↓
128D Face Embedding
    ↓
PostgreSQL + pgvector nearest-neighbor search
    ↓
Known person → latest summary overlay + TTS reminder
Unknown person → create user profile + start conversation session

Laptop Microphone
    ↓
faster-whisper local STT
    ↓
Conversation transcript
    ↓
OpenAI GPT summary
    ↓
conversations table
```

## AI Pipeline

### Responsible AI and Model Usage

- No custom model training or fine-tuning is used.
- Face embeddings are generated locally using `face_recognition`, based on dlib.
- Speech-to-text runs locally with `faster-whisper`, configured for Vietnamese.
- OpenAI GPT is used only to summarize transcripts into short Vietnamese memory notes.
- Face matching is deterministic nearest-neighbor search through pgvector with a configurable distance threshold.
- Every saved conversation keeps the original transcript and the generated summary for later review.
- The application should be used with explicit consent from participants in any real-world setting.

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python | Main application runtime |
| Video | OpenCV | Webcam capture, frame loop, real-time overlay |
| Face Recognition | face_recognition, dlib | Face detection and 128D face embeddings |
| Audio | sounddevice, soundfile | Microphone recording and temporary audio chunks |
| Speech | faster-whisper | Local Vietnamese speech-to-text |
| Text-to-Speech | pyttsx3 | Spoken memory reminders |
| Data | PostgreSQL, pgvector | Users, face embeddings, conversation history |
| AI | OpenAI API | Conversation summarization |
| Configuration | python-dotenv | Local `.env` configuration |
| Infrastructure | Docker Compose | Reproducible local PostgreSQL + pgvector setup |

## 🌟 Key Features

- 👤 Real-time face detection from laptop webcam.
- 🧬 128-dimensional face embedding extraction.
- 🔎 Approximate face lookup using pgvector cosine distance.
- 👥 Multiple faces supported in the same video frame.
- 🧠 Instant memory prompt for known people.
- 🆕 New-person registration when no match is found.
- 🎙️ Vietnamese microphone transcription with local `faster-whisper`.
- ✨ GPT-generated 1-2 sentence conversation summaries.
- 🗂️ Conversation history stored without overwriting previous meetings.
- 🔊 Optional spoken reminders through TTS.
- ⚡ `--fast` mode for smoother live demos.
- 🎯 `--accurate-detect` mode for better face detection sensitivity.
- 📝 Manual note shortcut for live demos when audio is noisy.

## 🔌 Core Modules

| Module | Purpose |
|---|---|
| `main.py` | Main webcam loop, session orchestration, keyboard controls |
| `face_module.py` | Face detection, embeddings, bounding boxes, overlay text |
| `audio_module.py` | Microphone recording, faster-whisper transcription, TTS speaker |
| `db_module.py` | PostgreSQL connection, pgvector lookup, user and conversation queries |
| `llm_module.py` | OpenAI GPT summarization with fallback behavior |
| `db/init.sql` | Database schema and pgvector index creation |
| `docker-compose.yml` | Local PostgreSQL service with pgvector enabled |

## 🗃️ Database Schema

| Table | Fields |
|---|---|
| `users` | `id`, `name`, `face_embedding vector(128)`, `first_seen`, `last_seen` |
| `conversations` | `id`, `user_id`, `transcript`, `summary`, `created_at` |

## 🚀 Installation & Usage

### Prerequisites

- Python 3.11+ or 3.12+
- Docker and Docker Compose
- Webcam and microphone
- An OpenAI API key
- On Windows, `face_recognition` may require either Visual Studio C++ Build Tools or the prebuilt `dlib-bin` package

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd An_App_For_Short_Term_Memory_Loss_People-
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Set at least:

```env
OPENAI_API_KEY=your_openai_api_key
```

Important local settings:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=memory_assistant
POSTGRES_USER=memory_user
POSTGRES_PASSWORD=memory_password
```

Never commit `.env` or expose `OPENAI_API_KEY`.

### 3. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

The database schema is initialized from `db/init.sql` when the Docker volume is first created.

### 4. Create Python environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If `dlib` fails to build on Windows, install a prebuilt binary first:

```bash
pip install dlib-bin
pip install face_recognition==1.3.0 --no-deps
pip install -r requirements.txt
```

### 5. Run the demo

Recommended live demo mode:

```bash
python main.py --fast --accurate-detect --auto-name-new-people
```

If CPU usage is high, disable live audio:

```bash
python main.py --fast --accurate-detect --auto-name-new-people --no-audio
```

Useful options:

```bash
python main.py --webcam-index 1
python main.py --no-audio
python main.py --no-tts
python main.py --auto-name-new-people
python main.py --fast
python main.py --accurate-detect
```

### Demo controls

| Key | Action |
|---|---|
| `q` | Quit and save active sessions |
| `e` | Manually end and save the current visible session |
| `m` | Add a manual note to the visible person |

## 🧪 Testing

Basic syntax check:

```bash
python -m py_compile main.py face_module.py audio_module.py db_module.py llm_module.py
```

Database health:

```bash
docker compose ps
```

Manual demo checklist:

- Start Docker Compose and confirm PostgreSQL is healthy.
- Run the webcam demo.
- Show a face to the camera.
- Confirm a new user profile is created.
- Speak a short Vietnamese conversation.
- Press `e` or leave the frame until the session ends.
- Return to the camera and confirm the saved summary appears.

## 🗺️ Pilot Roadmap

Phase 1 — Webcam prototype: stabilize webcam recognition, conversation capture, and memory overlay for one laptop setup.

Phase 2 — Better interaction: add speaker diarization, explicit consent flow, profile editing, and a conversation history screen.

Phase 3 — Smart-glasses prototype: connect the same backend logic to a wearable camera or mobile companion app.

Phase 4 — Caregiver support: allow trusted caregivers to review, correct, and export memory summaries with clear privacy controls.

## 🤝 Contributing

Contributions are welcome. Useful areas include:

- Faster and more reliable face detection.
- Better Vietnamese speech-to-text handling in noisy rooms.
- Speaker diarization for multi-person conversations.
- A consent-first onboarding and privacy UX.
- A review screen for editing saved names and summaries.
- Tests for database logic, summarization fallback, and session lifecycle.

Before contributing, please keep the assistive scope clear: this project should help users remember context, not classify health conditions or make medical claims.

## 👥 Maintainers

| Member | Role | Profile |
|---|---|---|
| Maintainers | Assistive AI · Prototype Engineering · Open Source | GitHub |

## 📄 License

This project is intended to be open source. Add the repository's chosen license text in `LICENSE`.

Built for a more supportive memory experience: calmer for users, clearer for caregivers, and easier to demonstrate in real time.
