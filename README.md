<div align="center">

# 🧠✨ Memory Assistant

### Recognize familiar faces · Remember conversations · Recall at the right moment

A smart-glasses demo designed to help people with short-term memory difficulties recognize who they are speaking with and recall their previous conversation.

`📷 Face Recognition` · `🎙️ Vietnamese STT` · `✨ GPT Summaries` · `🐘 PostgreSQL` · `🔎 pgvector`

</div>

---

## 🎬 Demo

![Memory Assistant demo](<docs/assets/Screen Recording 2026-07-18 at 14.11.30.gif>)

## 🌟 What can it do?

- 👤 Detect multiple faces using 128-dimensional face embeddings.
- 💬 Show one animated gradient bubble with a name, live transcript, and previous memory.
- 🪪 Replace “Familiar person” with the person’s name when it is clearly spoken.
- ⏱️ Show how long ago the previous conversation happened.
- 🎙️ Recognize Vietnamese speech with `faster-whisper`.
- 🛡️ Discard low-confidence audio instead of displaying guessed text.
- ✨ Ask GPT to summarize only when the transcript contains clear information.
- 🔊 Play a TTS reminder only when the same person returns.
- 🧠 Store faces and conversation history in PostgreSQL with pgvector.

## 🧭 How it works

```text
📷 Webcam → 👤 Face detection → 🔎 pgvector matching
                                      │
🎙️ Mac mic → 📝 Vietnamese text ──────┤
                                      ▼
                               ✨ GPT summary
                                      │
                                      ▼
                    💬 Name + time ago + memory bubble
```

## 🧰 Tech stack

| Area | Technology |
|---|---|
| 📷 Camera and UI | OpenCV + Pillow |
| 👤 Face recognition | `face_recognition` / dlib |
| 🎙️ Vietnamese speech | `faster-whisper` + `sounddevice` |
| ✨ Summaries | OpenAI API |
| 🔊 Voice reminders | `pyttsx3` |
| 🐘 Storage | PostgreSQL 16 + pgvector |
| 🐳 Local infrastructure | Docker Compose |

## 🚀 Quick start on macOS

### 1. Install the prerequisites

You will need:

- 🍎 macOS
- 🐍 Python 3.11
- 🐳 Docker Desktop running
- 🔑 An OpenAI API key

Install the required system libraries if they are missing:

```bash
brew install python@3.11 cmake portaudio libsndfile
```

### 2. Open the project directory

```bash
cd /Users/tus/Work/DETECH_APP/An_App_For_Short_Term_Memory_Loss_People-
```

### 3. Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the application

```bash
cp .env.example .env
```

Open `.env` and replace the following value with your real key:

```dotenv
OPENAI_API_KEY=sk-your-key-here
```

> 🔐 Never commit or share your `.env` file.

### 5. Start the database

```bash
docker compose up -d
docker compose ps
```

The `memory-assistant-postgres` service should report a `healthy` status.

### 6. Allow camera and microphone access

Open:

`System Settings → Privacy & Security → Camera / Microphone`

Enable access for the application used to run the project, such as `Terminal`, `Visual Studio Code`, or `Codex`.

### 7. Run the application 🎉

Recommended command for a Mac:

```bash
.venv/bin/python main.py --webcam-index 0 --fast --auto-name-new-people
```

The application is ready when the terminal prints:

```text
Demo đã sẵn sàng.
Đang dùng webcam index 0.
```

> These startup messages are in Vietnamese because the application UI and voice experience are configured for Vietnamese users.

## 💻 Run in Visual Studio Code

1. Open the project:

   ```bash
   code /Users/tus/Work/DETECH_APP/An_App_For_Short_Term_Memory_Loss_People-
   ```

2. Press `⌘ ⇧ P` → choose `Python: Select Interpreter` → select `.venv/bin/python`.
3. Open the integrated terminal with `` Ctrl + ` ``.
4. Start the database and application:

   ```bash
   docker compose up -d
   .venv/bin/python main.py --webcam-index 0 --fast --auto-name-new-people
   ```

> 🎥 If VS Code cannot access the camera or microphone, enable **Visual Studio Code** under `System Settings → Privacy & Security`, then restart VS Code.

## 🎮 Controls

| Key | Action |
|:---:|---|
| `q` | 👋 Exit and save active sessions |
| `e` | 💾 End the current session and create a summary |
| `m` | ✍️ Add a manual note |

## 🪄 Useful run modes

```bash
# Smoother performance on a laptop
.venv/bin/python main.py --fast --auto-name-new-people

# Improve face detection sensitivity
.venv/bin/python main.py --fast --accurate-detect --auto-name-new-people

# Select another camera
.venv/bin/python main.py --webcam-index 1 --fast --auto-name-new-people

# Disable microphone input or voice reminders
.venv/bin/python main.py --no-audio
.venv/bin/python main.py --no-tts
```

## 🩺 Troubleshooting

<details>
<summary><strong>📷 The camera does not open</strong></summary>

- Check the macOS Camera permission.
- Close Zoom, Teams, or another application that may be using the camera.
- Use `--webcam-index 0` to prioritize the Mac’s built-in camera.

</details>

<details>
<summary><strong>🎙️ Vietnamese speech is not recognized</strong></summary>

- Check the Microphone permission and select `MacBook Pro Microphone` in macOS.
- Speak close to the Mac in complete sentences and avoid loud external speakers.
- `STT_MIN_CONFIDENCE=0.8` favors missing uncertain speech over displaying hallucinated text.

</details>

<details>
<summary><strong>🐘 PostgreSQL does not connect</strong></summary>

```bash
docker compose up -d
docker compose ps
docker compose logs postgres
```

</details>

<details>
<summary><strong>✨ GPT does not work</strong></summary>

- Check `OPENAI_API_KEY` in `.env`.
- Do not wrap the key in quotes or add spaces around it.
- If GPT fails, the transcript is preserved but an unverified summary is not displayed.

</details>

## 🗂️ Project structure

```text
.
├── 🚀 main.py              # Camera, audio, database, and UI orchestration
├── 📷 face_module.py       # Face recognition and Unicode bubbles
├── 🎙️ audio_module.py      # Whisper STT and TTS
├── 🐘 db_module.py         # PostgreSQL and pgvector
├── ✨ llm_module.py        # GPT conversation summaries
├── 🐳 docker-compose.yml   # Local PostgreSQL service
├── 🗃️ db/init.sql          # Database schema
├── 🎬 docs/assets          # Demo GIF
├── 📦 requirements.txt
└── ⚙️ .env.example
```

## ⚙️ Important configuration

| Variable | Default | Purpose |
|---|:---:|---|
| `WEBCAM_INDEX` | `0` | Preferred camera |
| `FACE_DISTANCE_THRESHOLD` | `0.62` | L2 face-matching threshold |
| `FACE_ABSENCE_TIMEOUT_SECONDS` | `3` | Delay before ending a session |
| `WHISPER_LANGUAGE` | `vi` | Speech-recognition language |
| `AUDIO_CHUNK_SECONDS` | `2` | Audio chunk duration |
| `STT_MIN_CONFIDENCE` | `0.8` | Hallucinated-transcript filter |
| `TTS_ENABLED` | `true` | Voice reminders for returning people |

## 📝 Notes

- A single microphone cannot reliably separate multiple simultaneous speakers. A transcript is attached to the faces visible at that moment.
- Ending a session creates a new history entry instead of overwriting an older conversation.
- Face matching uses pgvector Euclidean/L2 distance (`<->`).
- To delete all test data, run `docker compose down -v` — this action cannot be undone.

---

<div align="center">

Made with 🧠, 🎙️, ✨, and a slightly forgetful webcam.

</div>
