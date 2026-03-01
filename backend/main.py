"""
Axonara – FastAPI Backend
==========================
REST API that orchestrates:
  • Text extraction
  • Summarization & simplification
  • Key-concept extraction
  • Mind-map generation
  • Flashcard generation
  • Cognitive-load detection
"""

import os
import sys
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the backend package is importable when run from the project root
sys.path.insert(0, os.path.dirname(__file__))

from nlp_engine import (
    extract_text,
    summarize,
    simplify,
    extract_key_concepts,
    generate_mind_map,
    render_mind_map_image,
    generate_flashcards,
)
from cognitive_model import cognitive_model

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Axonara API",
    description="AI-powered adaptive learning platform for neurodiverse learners",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    raw_text: Optional[str] = None
    typing_speed: float = 50.0
    pause_duration: float = 3.0
    edit_frequency: float = 5.0


class FlashCard(BaseModel):
    front: str
    back: str


class MindMap(BaseModel):
    nodes: list[str]
    edges: list[dict]


class CognitiveStatus(BaseModel):
    overload: bool
    overload_score: float
    status: str
    recommendation: str


class ProcessResponse(BaseModel):
    original_text: str
    summary: str
    simplified: str
    key_concepts: list[str]
    mind_map: MindMap
    mind_map_image_base64: str
    flashcards: list[FlashCard]
    cognitive: CognitiveStatus
    adapted_output: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Axonara API is running 🧠"}


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    typing_speed: float = Form(50.0),
    pause_duration: float = Form(3.0),
    edit_frequency: float = Form(5.0),
):
    """
    Main pipeline endpoint.

    Accepts either a file upload (PDF / TXT) or pasted raw text, plus
    simulated cognitive-load inputs. Returns the full adaptive learning
    package.
    """
    # 1. Extract text
    file_bytes = None
    filename = ""
    if file is not None:
        file_bytes = await file.read()
        filename = file.filename or ""

    text = extract_text(file_bytes=file_bytes, raw_text=raw_text, filename=filename)
    if not text:
        return ProcessResponse(
            original_text="",
            summary="No text provided.",
            simplified="No text provided.",
            key_concepts=[],
            mind_map=MindMap(nodes=[], edges=[]),
            mind_map_image_base64="",
            flashcards=[],
            cognitive=CognitiveStatus(
                overload=False, overload_score=0, status="normal",
                recommendation="Please upload a document or paste text."
            ),
            adapted_output="No text provided.",
        )

    # 2. Summarize
    summary = summarize(text)

    # 3. Simplify
    simplified = simplify(text)

    # 4. Key concepts
    concepts = extract_key_concepts(text)

    # 5. Mind map
    title = concepts[0] if concepts else "Main Topic"
    mind_map_data = generate_mind_map(title, concepts)
    mind_map_img = render_mind_map_image(mind_map_data)
    mind_map_b64 = base64.b64encode(mind_map_img).decode()

    # 6. Flashcards
    flashcards = generate_flashcards(text)

    # 7. Cognitive load
    cognitive = cognitive_model.predict(typing_speed, pause_duration, edit_frequency)

    # 8. Adaptive output
    if cognitive["overload"]:
        adapted_output = (
            "⚠️ Cognitive overload detected — presenting simplified content:\n\n"
            + simplified
        )
    else:
        adapted_output = (
            "✅ Cognitive load normal — presenting full summary:\n\n"
            + summary
        )

    return ProcessResponse(
        original_text=text[:3000],  # truncate for response size
        summary=summary,
        simplified=simplified,
        key_concepts=concepts,
        mind_map=MindMap(**mind_map_data),
        mind_map_image_base64=mind_map_b64,
        flashcards=[FlashCard(**fc) for fc in flashcards],
        cognitive=CognitiveStatus(**cognitive),
        adapted_output=adapted_output,
    )


@app.post("/cognitive")
def check_cognitive_load(
    typing_speed: float = Form(50.0),
    pause_duration: float = Form(3.0),
    edit_frequency: float = Form(5.0),
):
    """Standalone cognitive-load check."""
    return cognitive_model.predict(typing_speed, pause_duration, edit_frequency)


# ---------------------------------------------------------------------------
# Run with: uvicorn backend.main:app --reload
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
