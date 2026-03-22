# indexing subpackage — PDF parsing, chunking, enrichment, pipeline
try:
    from .pipeline import IndexingPipeline  # noqa: F401
except ImportError:
    pass

from .chunker import chunk_text_docs, extract_tables_and_text  # noqa: F401
from .pdf_parser import clean_markdown, load_pdf_with_docling  # noqa: F401
from .image_enricher import (  # noqa: F401
    enrich_images,
    replace_images_with_placeholders,
    restore_image_placeholders,
)
