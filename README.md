# 🧠 Axonara AI — Adaptive Learning Platform

**AI-powered adaptive learning prototype designed to help neurodiverse learners by detecting cognitive overload and transforming educational content into easier learning formats.**

Built for hackathon — powered by HuggingFace Transformers, Sentence-Transformers, Scikit-learn & NetworkX.

---

## ✨ Features

- 📄 **Content Upload** — Upload PDF or paste text directly
- 📝 **AI Summarization** — Two-stage summarization (extractive + abstractive using DistilBART)
- 🔑 **Key Concept Extraction** — Sentence-Transformer powered topic extraction
- 🗺️ **Mind Map Generation** — Auto-generated visual mind maps using NetworkX
- 🃏 **Flashcard Generation** — Automatic Q&A flashcards from key sentences
- 🧠 **Cognitive Overload Detection** — Simulated learner behavior analysis using Logistic Regression
- 🔄 **Adaptive Output** — Content automatically simplifies when overload is detected

---

## 📁 Project Structure

```
├── backend/
│   ├── cognitive_model.py   # Cognitive load detection (Logistic Regression)
│   ├── nlp_engine.py        # Summarization, concepts, mind map, flashcards
│   └── main.py              # FastAPI REST API
├── frontend/
│   └── app.py               # Gradio web interface
├── data/                    # For uploaded files
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI |
| NLP / Summarization | HuggingFace Transformers (DistilBART) |
| Key Concept Extraction | Sentence-Transformers (all-MiniLM-L6-v2) |
| Cognitive Load Detection | Scikit-learn (Logistic Regression) |
| Mind Map Visualization | NetworkX + Matplotlib |
| Frontend | Gradio |

---

## 🚀 How to Run (Step by Step)

### Prerequisites

- **Python 3.10+** installed ([Download Python](https://www.python.org/downloads/))
- **Git** installed ([Download Git](https://git-scm.com/downloads))
- **~2 GB disk space** for AI model downloads (first run only)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AnantSinghTanwar-ux/AxonaraAI.git
cd AxonaraAI
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages: FastAPI, Transformers, Sentence-Transformers, Scikit-learn, Gradio, NetworkX, PyPDF2, etc.

### Step 4: Run the Application

```bash
python frontend/app.py
```

### Step 5: Open in Browser

Go to **http://localhost:7860** or **http://127.0.0.1:7860**

> ⏳ **First run note:** The app will download two AI models (~1.2 GB total). This happens once — after that, models are cached locally.

---

## 📖 How to Use

1. **Upload a PDF** or **paste text** into the text box
2. **Adjust the cognitive load sliders** (simulates learner behavior):
   - **Typing Speed** — Lower = higher cognitive load
   - **Pause Duration** — Higher = higher cognitive load
   - **Edit Frequency** — Higher = higher cognitive load
3. Click **🚀 Process**
4. View results across tabs:
   - **Adaptive Output** — Summary adjusted based on cognitive load
   - **Mind Map** — Visual concept map
   - **Flashcards** — Auto-generated Q&A cards
   - **Cognitive Status** — Overload detection results
   - **Full Summary** — Complete AI summary

---

## 🧠 How Cognitive Load Detection Works

The system uses a **Logistic Regression** model trained on synthetic behavioral data:

| Signal | Normal Range | Overload Range |
|--------|-------------|---------------|
| Typing Speed (wpm) | ~60 | ~25 |
| Pause Duration (sec) | ~2 | ~8 |
| Edit Frequency (/min) | ~3 | ~10 |

- If **overload score > 55%** → content is **simplified** (bullet points)
- If **overload score ≤ 55%** → **full summary** is displayed

---

## 🔗 API (Optional)

You can also run the FastAPI backend separately:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/process` | Full pipeline (upload file + cognitive inputs) |
| POST | `/cognitive` | Standalone cognitive load check |

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `port 7860 already in use` | Kill the old process: `netstat -ano \| findstr :7860` then `taskkill /F /PID <PID>` |
| `ModuleNotFoundError` | Make sure you activated the venv and ran `pip install -r requirements.txt` |
| App loads but processing is slow | First run downloads models (~1.2 GB). Subsequent runs are faster (15-30 sec). |
| PDF won't process | Make sure the PDF has selectable text (not scanned images) |
| Browser shows "can't reach site" | Use `http://localhost:7860` not `http://0.0.0.0:7860` |

---

## 📄 License

This project is a hackathon prototype built for educational purposes.

---

**Made with ❤️ by Anant Singh Tanwar**
