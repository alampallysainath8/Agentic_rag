import base64
import logging
import re
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_PAGE_NUM_RE = re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$")
_PAGE_LABEL_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)


def clean_markdown(markdown: str) -> str:
    """Strip page numbers, orphaned headers, and collapse excess blank lines."""
    lines = markdown.split("\n")

    filtered = [
        ln for ln in lines
        if not _PAGE_NUM_RE.match(ln.strip()) and not _PAGE_LABEL_RE.match(ln.strip())
    ]

    result = []
    n = len(filtered)
    for i, line in enumerate(filtered):
        if line.strip().startswith("#"):
            next_content = next(
                (filtered[j].strip() for j in range(i + 1, n) if filtered[j].strip()),
                None,
            )
            if next_content is None or next_content.startswith("#"):
                continue
        result.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()


def load_pdf_with_docling(filepath: str, scratch_dir: str = "./indexing/scratch_images") -> Tuple[str, List[str]]:
    """
    Convert a PDF to markdown using Docling and extract all embedded images.

    Returns:
        markdown  — full document as markdown (image refs use REFERENCED mode)
        img_paths — list of absolute paths to extracted PNG files
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc import ImageRefMode, PictureItem

    path = Path(filepath)
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(path)
    doc = result.document

    img_paths: List[str] = []
    counter = 0
    for element, _ in doc.iterate_items():
        if isinstance(element, PictureItem):
            counter += 1
            img_file = scratch / f"{path.stem}_img_{counter}.png"
            with open(img_file, "wb") as f:
                element.get_image(doc).save(f, "PNG")
            img_paths.append(str(img_file))

    md_file = scratch / f"{path.stem}.md"
    doc.save_as_markdown(md_file, image_mode=ImageRefMode.REFERENCED)
    markdown = md_file.read_text(encoding="utf-8")

    logger.info("Docling: '%s' — %d images extracted", path.name, counter)
    return markdown, img_paths



