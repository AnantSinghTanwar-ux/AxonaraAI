"""
Axonara - NLP Engine
=====================
Handles:
  • Text extraction from PDF / raw text
  • Summarization  (HuggingFace BART)
  • Key-concept extraction  (Sentence-Transformers + cosine similarity)
  • Mind-map graph generation  (NetworkX)
  • Flashcard generation
  • Simplified explanation generation
"""

import io
import re
import textwrap
from typing import List, Tuple

import networkx as nx
import numpy as np

# PDF parsing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None  # graceful fallback – user must install

# HuggingFace – use model classes directly for max compatibility
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Sentence-Transformers
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoids reload on every request)
# ---------------------------------------------------------------------------

_bart_model = None
_bart_tokenizer = None
_sentence_model = None

# DistilBART is ~3x faster than full BART on CPU (~300MB vs 1.6GB)
_BART_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Maximum characters we'll feed to the NLP pipeline (keeps processing fast)
_MAX_INPUT_CHARS = 8000


def _get_bart():
    """Load DistilBART model + tokenizer (lazy, once)."""
    global _bart_model, _bart_tokenizer
    if _bart_model is None:
        print("[Axonara] Loading DistilBART model – first run downloads ~1.2GB…")
        _bart_tokenizer = AutoTokenizer.from_pretrained(_BART_MODEL_NAME)
        _bart_model = AutoModelForSeq2SeqLM.from_pretrained(_BART_MODEL_NAME)
        _bart_model.eval()
        print("[Axonara] DistilBART model loaded ✓")
    return _bart_model, _bart_tokenizer


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        print("[Axonara] Loading Sentence-Transformer model…")
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Axonara] Sentence-Transformer loaded ✓")
    return _sentence_model


def _truncate_text(text: str) -> str:
    """Keep only the first N chars to avoid very long processing times."""
    if len(text) > _MAX_INPUT_CHARS:
        # Cut at last sentence boundary within limit
        truncated = text[:_MAX_INPUT_CHARS]
        last_period = truncated.rfind(".")
        if last_period > _MAX_INPUT_CHARS // 2:
            truncated = truncated[:last_period + 1]
        return truncated
    return text


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from an in-memory PDF."""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required for PDF parsing. pip install PyPDF2")
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text(file_bytes: bytes | None = None, raw_text: str | None = None, filename: str = "") -> str:
    """Return clean text from either a PDF upload or pasted text.
    Prioritises uploaded file over raw text."""
    # Prefer uploaded file when available
    if file_bytes:
        try:
            if filename.lower().endswith(".pdf"):
                pdf_text = extract_text_from_pdf(file_bytes)
                if pdf_text and pdf_text.strip():
                    return pdf_text.strip()
            else:
                decoded = file_bytes.decode("utf-8", errors="ignore").strip()
                if decoded:
                    return decoded
        except Exception as e:
            print(f"[Axonara] File extraction failed: {e}")
    # Fall back to pasted text
    if raw_text and raw_text.strip():
        return raw_text.strip()
    return ""


# ---------------------------------------------------------------------------
# 2. Summarization
# ---------------------------------------------------------------------------

def _extractive_summary(text: str, n_sentences: int = 5) -> str:
    """
    Fast extractive summary: pick the top-N most representative sentences
    using Sentence-Transformer cosine similarity to the full document.
    No heavy generative model needed – runs in ~1 second.
    """
    model = _get_sentence_model()
    sentences = _split_sentences(text)
    if len(sentences) <= n_sentences:
        return text

    # Only embed the first 200 sentences max for speed
    sentences = sentences[:200]
    doc_emb = model.encode(text[:5000], convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(doc_emb, sent_embs)[0].cpu().numpy()
    top_idx = sorted(np.argsort(scores)[-n_sentences:])
    return " ".join(sentences[i] for i in top_idx)


def summarize(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """
    Two-stage summarization:
      1. Extractive pass (fast) to pick the best sentences
      2. Abstractive pass with DistilBART on the extracted text (single chunk)
    This keeps processing under ~15-20 seconds even for large documents.
    """
    if not text:
        return ""
    import torch

    text = _truncate_text(text)

    # Stage 1 – extractive (instant)
    extracted = _extractive_summary(text, n_sentences=8)
    print(f"[Axonara] Extractive summary: {len(extracted)} chars")

    # Stage 2 – abstractive with DistilBART (single pass, fast)
    model, tokenizer = _get_bart()
    inputs = tokenizer(
        extracted,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=2,          # 2 beams instead of 4 = much faster
            length_penalty=1.0,
            early_stopping=True,
        )
    decoded = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return decoded


def simplify(text: str) -> str:
    """
    Produce a simplified version of *text* suitable for overloaded learners.
    Uses extractive summary converted to bullet points (very fast).
    """
    if not text:
        return ""
    text = _truncate_text(text)
    extracted = _extractive_summary(text, n_sentences=5)
    sentences = _split_sentences(extracted)
    bullets = "\n".join(f"• {s.strip()}" for s in sentences if s.strip())
    return bullets


# ---------------------------------------------------------------------------
# 3. Key concept extraction
# ---------------------------------------------------------------------------

def extract_key_concepts(text: str, top_n: int = 8) -> List[str]:
    """
    Extract the most representative noun-phrase-like concepts.
    Only processes the first 150 sentences for speed.
    """
    if not text:
        return []
    text = _truncate_text(text)
    model = _get_sentence_model()
    sentences = _split_sentences(text)[:150]  # cap for speed
    if not sentences:
        return []

    doc_embedding = model.encode(text[:5000], convert_to_tensor=True)
    sent_embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(doc_embedding, sent_embeddings)[0].cpu().numpy()

    ranked_idx = np.argsort(scores)[::-1]

    concepts: list[str] = []
    seen: set[str] = set()
    for i in ranked_idx:
        phrase = _extract_noun_phrase(sentences[i])
        key = phrase.lower()
        if key not in seen and len(phrase) > 3:
            seen.add(key)
            concepts.append(phrase)
        if len(concepts) >= top_n:
            break

    return concepts


# ---------------------------------------------------------------------------
# 4. Mind map generation
# ---------------------------------------------------------------------------

def generate_mind_map(title: str, concepts: List[str]) -> dict:
    """
    Build a NetworkX graph and return serialisable edge-list + node list.
    Central node = title, spokes = key concepts.
    """
    G = nx.Graph()
    center = title or "Main Topic"
    G.add_node(center, role="center")
    for concept in concepts:
        G.add_node(concept, role="concept")
        G.add_edge(center, concept)

    # Also link closely-related concepts (cosine sim > 0.5)
    if len(concepts) >= 2:
        model = _get_sentence_model()
        embs = model.encode(concepts, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embs, embs).cpu().numpy()
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if sim_matrix[i][j] > 0.5:
                    G.add_edge(concepts[i], concepts[j])

    return {
        "nodes": list(G.nodes),
        "edges": [{"source": u, "target": v} for u, v in G.edges],
    }


def render_mind_map_image(mind_map: dict) -> bytes:
    """Render the mind map as a PNG image and return raw bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = nx.Graph()
    for n in mind_map["nodes"]:
        G.add_node(n)
    for e in mind_map["edges"]:
        G.add_edge(e["source"], e["target"])

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Determine center node (first in the list)
    center = mind_map["nodes"][0] if mind_map["nodes"] else None
    node_colors = [
        "#4A90D9" if n == center else "#7EC8E3" for n in G.nodes
    ]
    node_sizes = [
        3000 if n == center else 1800 for n in G.nodes
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="#333")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    ax.set_title("Axonara – Mind Map", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# 5. Flashcard generation
# ---------------------------------------------------------------------------

def generate_flashcards(text: str, max_cards: int = 8) -> List[dict]:
    """
    Create Q/A flashcards from the most informative sentences.
    Front = question derived from the sentence.
    Back  = the original sentence.
    """
    if not text:
        return []
    text = _truncate_text(text)
    model = _get_sentence_model()
    sentences = _split_sentences(text)[:150]  # cap for speed
    if not sentences:
        return []

    doc_emb = model.encode(text[:5000], convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(doc_emb, sent_embs)[0].cpu().numpy()
    ranked = np.argsort(scores)[::-1]

    cards: list[dict] = []
    seen: set[str] = set()
    for i in ranked:
        sent = sentences[i].strip()
        if len(sent) < 15 or sent.lower() in seen:
            continue
        seen.add(sent.lower())
        question = _sentence_to_question(sent)
        cards.append({"front": question, "back": sent})
        if len(cards) >= max_cards:
            break
    return cards


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = 2500) -> list[str]:
    sentences = _split_sentences(text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_chars]]


def _split_sentences(text: str) -> list[str]:
    """Basic sentence splitter."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _extract_noun_phrase(sentence: str) -> str:
    """Return a short key-phrase from a sentence (heuristic)."""
    # Take the first ~6 significant words after removing filler
    words = sentence.split()
    filler = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "in", "to", "it", "that", "this", "for", "on", "with"}
    sig = [w for w in words if w.lower().strip(".,;:!?") not in filler]
    phrase = " ".join(sig[:6])
    # Clean trailing punctuation
    return phrase.strip(".,;:!?\"'")


def _sentence_to_question(sentence: str) -> str:
    """Convert a declarative sentence into a simple question prompt."""
    sentence = sentence.strip().rstrip(".")
    if len(sentence) > 90:
        sentence = sentence[:90] + "…"
    return f"What do you know about: {sentence}?"


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Supervised learning uses labeled "
        "datasets to train algorithms. Unsupervised learning finds hidden patterns "
        "in unlabeled data. Reinforcement learning trains agents through rewards "
        "and penalties. Deep learning uses neural networks with many layers to "
        "model complex patterns in large amounts of data. Natural language "
        "processing enables computers to understand and generate human language."
    )
    print("--- Summary ---")
    print(summarize(sample))
    print("\n--- Key Concepts ---")
    print(extract_key_concepts(sample))
    print("\n--- Mind Map ---")
    print(generate_mind_map("Machine Learning", extract_key_concepts(sample)))
    print("\n--- Flashcards ---")
    for fc in generate_flashcards(sample):
        print(f"  Q: {fc['front']}")
        print(f"  A: {fc['back']}\n")
