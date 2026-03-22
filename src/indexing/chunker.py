import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


def extract_tables_and_text(
    markdown: str, source: str
) -> Tuple[List[Document], List[Document]]:
    """
    Parse *markdown* line-by-line.

    Lines starting with '|' are grouped into atomic table Documents.
    All other lines are collected into a single text Document.

    Returns
    -------
    text_docs  : list with 0-1 Document(type="text")
    table_docs : list of Document(type="table"), one per contiguous table block
    """
    lines = markdown.split("\n")
    text_lines: List[str] = []
    table_docs: List[Document] = []
    current_table: List[str] = []
    in_table = False

    for line in lines:
        if line.strip().startswith("|"):
            in_table = True
            current_table.append(line)
        else:
            if in_table:
                table_docs.append(
                    Document(
                        page_content="\n".join(current_table),
                        metadata={"source": source, "type": "table"},
                    )
                )
                current_table = []
                in_table = False
            text_lines.append(line)

    # flush trailing table
    if current_table:
        table_docs.append(
            Document(
                page_content="\n".join(current_table),
                metadata={"source": source, "type": "table"},
            )
        )

    text_content = "\n".join(text_lines).strip()
    text_docs = (
        [Document(page_content=text_content, metadata={"source": source, "type": "text"})]
        if text_content
        else []
    )

    logger.debug(
        "extract_tables_and_text: %d text doc(s), %d table doc(s)",
        len(text_docs),
        len(table_docs),
    )
    return text_docs, table_docs


def chunk_text_docs(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Two-pass chunking for text Documents.

    Pass 1 - MarkdownHeaderTextSplitter
        Splits on #, ##, ### headers, preserving header metadata.
    Pass 2 - RecursiveCharacterTextSplitter
        Further splits any split that still exceeds *chunk_size*.

    Table Documents are passed through unchanged.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result: List[Document] = []
    for doc in docs:
        if doc.metadata.get("type") == "table":
            result.append(doc)
            continue

        header_splits = md_splitter.split_text(doc.page_content)
        for split in header_splits:
            merged_meta = {**doc.metadata, **split.metadata}
            sub_chunks = rec_splitter.create_documents(
                [split.page_content], metadatas=[merged_meta]
            )
            result.extend(sub_chunks)

    logger.debug("chunk_text_docs: produced %d chunk(s)", len(result))
    return result
