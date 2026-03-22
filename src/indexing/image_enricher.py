import base64
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_IMG_REF_RE = re.compile(r"!\[.*?\]\(.*?\)")
_IMG_TOKEN_RE = re.compile(r"\{\{IMG_(\d+)\}\}")

_VISION_SYSTEM = SystemMessage(content=(
    "You are a document analysis assistant. "
    "Given an image from a PDF document and its surrounding text context, extract:\n"
    "- Key visual elements (charts, diagrams, figures, tables rendered as images)\n"
    "- All numeric or textual values visible\n"
    "- Any trends, labels, or structural insights\n"
    "Return a concise, structured plain-text description — no markdown headers."
))


def _describe_image(vision_model: str, api_key: str, img_path: str, context: str) -> str:
    from langchain_groq import ChatGroq
    client = ChatGroq(model=vision_model, api_key=api_key, temperature=0.2)
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(img_path).suffix.lstrip(".")
    message = HumanMessage(content=[
        {"type": "text", "text": f"Surrounding context:\n{context}\n\nDescribe this image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
    ])
    response = client.invoke([_VISION_SYSTEM, message])
    return response.content.strip()


def _extract_context(markdown: str, token: str, window: int = 300) -> str:
    """Return up to *window* chars of text before and after *token* in *markdown*."""
    idx = markdown.find(token)
    if idx == -1:
        return ""
    before = markdown[max(0, idx - window): idx].strip()
    after  = markdown[idx + len(token): idx + len(token) + window].strip()
    return "\n".join(p for p in [before, after] if p)


def replace_images_with_placeholders(
    markdown: str,
    img_paths: List[str],
) -> Tuple[str, Dict[str, dict]]:
    """
    Step 1 — replace every ``![alt](ref)`` in *markdown* with a ``{{IMG_N}}``
    token immediately, before any chunking or enrichment.

    Returns
    -------
    tokenised_md  : markdown with image refs replaced by ``{{IMG_N}}`` tokens
    ph_map        : ``{token: {"img_path": str | None, "description": None}}``
    """
    ph_map: Dict[str, dict] = {}
    counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal counter
        key = f"{{{{IMG_{counter}}}}}"
        ph_map[key] = {
            "img_path": img_paths[counter] if counter < len(img_paths) else None,
            "description": None,
        }
        counter += 1
        return f"\n\n{key}\n\n"

    tokenised_md = re.sub(_IMG_REF_RE, _replace, markdown)
    logger.debug("replace_images_with_placeholders: %d token(s) inserted", counter)
    return tokenised_md, ph_map


def enrich_images(
    ph_map: Dict[str, dict],
    markdown: str,
    enrich_images: bool,
    vision_model: str,
    api_key: str,
) -> Dict[str, dict]:
    """
    Step 2 — fill ``ph_map[key]["description"]`` for every placeholder.

    * ``enrich_images=True``  → call the LLM vision model per image
    * ``enrich_images=False`` → use a short ``[Image N: <filename>]`` label

    Context for the vision model is extracted from *markdown* (the tokenised
    version) so headings / captions near the token are preserved.
    """
    for n, (key, value) in enumerate(ph_map.items(), start=1):
        img_path = value["img_path"]
        name = Path(img_path).name if img_path else f"img_{n}"

        if not enrich_images:
            value["description"] = f"[Image {n}: {name}]"
            continue

        if not img_path or not os.path.exists(img_path):
            logger.warning("Image file not found for %s — using label.", key)
            value["description"] = f"[Image {n}: file not found]"
            continue

        context = _extract_context(markdown, key)
        try:
            value["description"] = _describe_image(vision_model, api_key, img_path, context)
            logger.debug("Enriched %s (%s)", key, name)
        except Exception as exc:
            logger.warning("Enrichment failed for %s: %s — using label.", key, exc)
            value["description"] = f"[Image {n}: description unavailable]"

    return ph_map


def restore_image_placeholders(
    chunks: List[Document],
    ph_map: Dict[str, dict],
) -> List[Document]:
    """
    Step 4 — replace ``{{IMG_N}}`` tokens in the chunked Documents.

    * chunk is **only** a token  → becomes a dedicated image Document
    * token appears **inline**   → description is spliced in-place
    """
    if not ph_map:
        return chunks

    restored: List[Document] = []
    for chunk in chunks:
        content = chunk.page_content
        stripped = content.strip()

        if stripped in ph_map:
            desc = ph_map[stripped]["description"] or stripped
            restored.append(Document(
                page_content=desc,
                metadata={**chunk.metadata, "type": "image"},
            ))
        else:
            for key, val in ph_map.items():
                if key in content:
                    content = content.replace(key, val["description"] or key)
            restored.append(Document(page_content=content, metadata=chunk.metadata))

    return restored
