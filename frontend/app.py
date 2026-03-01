"""
Axonara – Gradio Frontend
===========================
Interactive UI that lets users:
  • Upload a PDF or paste text
  • Adjust simulated cognitive-load inputs
  • View: summary, mind map, flashcards, cognitive status, adaptive output
"""

import os
import sys
import io
import base64
import tempfile

import gradio as gr

# Allow imports from the backend package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

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
# Core processing function called by Gradio
# ---------------------------------------------------------------------------

def process_input(
    file,
    raw_text: str,
    typing_speed: float,
    pause_duration: float,
    edit_frequency: float,
):
    """Run the full Axonara pipeline and return all outputs."""

    # ---- 1. Extract text ---------------------------------------------------
    file_bytes = None
    filename = ""
    if file is not None:
        filepath = file if isinstance(file, str) else file.name
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            file_bytes = f.read()

    text = extract_text(file_bytes=file_bytes, raw_text=raw_text, filename=filename)

    if not text or len(text.strip()) < 20:
        empty = "⚠️ Please upload a document or paste some text (at least a few sentences).\n\nTip: If you uploaded a PDF, make sure it contains selectable text (not scanned images)."
        return empty, None, empty, empty, empty

    try:
        # ---- 2. Summarize --------------------------------------------------
        summary = summarize(text)

        # ---- 3. Simplify ---------------------------------------------------
        simplified = simplify(text)

        # ---- 4. Key concepts -----------------------------------------------
        concepts = extract_key_concepts(text)

        # ---- 5. Mind map ---------------------------------------------------
        title = concepts[0] if concepts else "Main Topic"
        mind_map_data = generate_mind_map(title, concepts)
        mind_map_bytes = render_mind_map_image(mind_map_data)
    except Exception as e:
        err = f"⚠️ Processing error: {e}\n\nThe text was extracted ({len(text)} chars) but NLP processing failed. Try pasting shorter text or a different document."
        return err, None, err, err, err

    # Write to temp file for Gradio Image component
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(mind_map_bytes)
    tmp.flush()
    mind_map_path = tmp.name
    tmp.close()

    # ---- 6. Flashcards -----------------------------------------------------
    flashcards = generate_flashcards(text)
    flashcard_text = ""
    for i, fc in enumerate(flashcards, 1):
        flashcard_text += f"### Card {i}\n"
        flashcard_text += f"**Q:** {fc['front']}\n\n"
        flashcard_text += f"**A:** {fc['back']}\n\n---\n\n"
    if not flashcard_text:
        flashcard_text = "No flashcards generated."

    # ---- 7. Cognitive load -------------------------------------------------
    cognitive = cognitive_model.predict(typing_speed, pause_duration, edit_frequency)

    status_icon = "🔴" if cognitive["overload"] else "🟢"
    cognitive_text = (
        f"## Cognitive Load Status: {status_icon} {cognitive['status'].upper()}\n\n"
        f"**Overload Score:** {cognitive['overload_score']:.2%}\n\n"
        f"**Recommendation:** {cognitive['recommendation']}\n\n"
        f"---\n"
        f"*Inputs used — Typing speed: {typing_speed} wpm, "
        f"Pause duration: {pause_duration}s, "
        f"Edit frequency: {edit_frequency}/min*"
    )

    # ---- 8. Adaptive output ------------------------------------------------
    if cognitive["overload"]:
        adapted = (
            "## ⚠️ Simplified Content (overload detected)\n\n"
            + simplified
            + "\n\n---\n*Content has been simplified to reduce cognitive load.*"
        )
    else:
        adapted = (
            "## ✅ Standard Summary\n\n"
            + summary
            + "\n\n### Key Concepts\n"
            + ", ".join(concepts)
            + "\n\n---\n*Learner is coping well — full content displayed.*"
        )

    return adapted, mind_map_path, flashcard_text, cognitive_text, summary


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Axonara – Adaptive Learning Platform",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 🧠 Axonara – AI-Powered Adaptive Learning
        **Helping neurodiverse learners by detecting cognitive overload and transforming content into easier learning formats.**

        Upload a PDF or paste text below, adjust the simulated cognitive-load sliders, then click **Process**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📄 Upload Document (PDF or TXT)",
                file_types=[".pdf", ".txt"],
            )
            text_input = gr.Textbox(
                label="✏️ Or paste text here",
                lines=8,
                placeholder="Paste educational content here…",
            )

            gr.Markdown("### 🎛️ Simulated Cognitive-Load Inputs")
            typing_speed = gr.Slider(
                10, 100, value=50, step=1,
                label="Typing Speed (words/min)",
                info="Lower = higher load",
            )
            pause_duration = gr.Slider(
                0.5, 15, value=3.0, step=0.5,
                label="Pause Duration (seconds)",
                info="Higher = higher load",
            )
            edit_frequency = gr.Slider(
                1, 20, value=5, step=1,
                label="Edit Frequency (edits/min)",
                info="Higher = higher load",
            )

            process_btn = gr.Button("🚀 Process", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📝 Adaptive Output"):
                    adaptive_output = gr.Markdown(label="Adaptive Output")

                with gr.TabItem("🗺️ Mind Map"):
                    mind_map_output = gr.Image(label="Mind Map", type="filepath")

                with gr.TabItem("🃏 Flashcards"):
                    flashcard_output = gr.Markdown(label="Flashcards")

                with gr.TabItem("🧠 Cognitive Status"):
                    cognitive_output = gr.Markdown(label="Cognitive Load")

                with gr.TabItem("📋 Full Summary"):
                    summary_output = gr.Markdown(label="Summary")

    process_btn.click(
        fn=process_input,
        inputs=[file_input, text_input, typing_speed, pause_duration, edit_frequency],
        outputs=[adaptive_output, mind_map_output, flashcard_output, cognitive_output, summary_output],
    )

    gr.Markdown(
        """
        ---
        *Axonara Prototype — Built for Hackathon. Powered by HuggingFace Transformers, Sentence-Transformers, Scikit-learn & NetworkX.*
        """
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
