"""Render each PDF page once as a grayscale PNG plus one compact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image


# Rendering/output settings.
DEFAULT_DPI = 200
DEFAULT_OUTPUT_DIR = "output/pages"
RENDER_GRAYSCALE = True
DRAW_ANNOTATIONS = True
PNG_COMPRESSION_LEVEL = 3
PAGE_NUMBER_DIGITS = 4

# Keep final and atomic-temp filename components comfortably Windows-safe.
MAX_INTERMEDIATE_FILENAME_CHARS = 96
INTERMEDIATE_HASH_CHARS = 16
ATOMIC_TEMP_FILENAME_OVERHEAD = 5


def validate_pdf_path(pdf_path: Path) -> Path:
    """Require an existing PDF file and return its expanded path."""
    pdf_path = pdf_path.expanduser()
    if not pdf_path.is_file():
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file is not a PDF: {pdf_path}")
    return pdf_path


def validate_dpi(dpi: int) -> int:
    """Require a positive PDF render resolution in DPI."""
    if dpi <= 0:
        raise ValueError(f"DPI must be greater than zero, received: {dpi}")
    return dpi


def ensure_output_dir(output_dir: Path) -> Path:
    """Create the image output directory and return its expanded path."""
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def bounded_intermediate_pdf_stem(pdf_path: Path, output_suffix: str) -> str:
    """Create a short, safe PDF stem with a hash if truncation is needed.

    Reserve space for the requested output suffix and temporary filename.
    """
    original = pdf_path.stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._-") or "report"
    # Reserve space for both the output suffix and atomic-write temporary name.
    limit = (
        MAX_INTERMEDIATE_FILENAME_CHARS
        - len(output_suffix)
        - ATOMIC_TEMP_FILENAME_OVERHEAD
    )
    if limit <= INTERMEDIATE_HASH_CHARS + 1:
        raise ValueError("Intermediate filename cap is too small for its suffix.")
    if len(safe) <= limit:
        return safe

    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:INTERMEDIATE_HASH_CHARS]
    return f"{safe[:limit - INTERMEDIATE_HASH_CHARS - 1]}_{digest}"


def make_page_image_path(
    pdf_path: Path,
    output_dir: Path,
    page_number: int,
) -> Path:
    """Build a page-numbered output PNG path using a bounded source stem."""
    suffix = f"_page_{page_number:0{PAGE_NUMBER_DIGITS}d}.png"
    return output_dir / f"{bounded_intermediate_pdf_stem(pdf_path, suffix)}{suffix}"


def make_manifest_path(pdf_path: Path, output_dir: Path) -> Path:
    """Build the output JSON manifest path for one PDF."""
    suffix = "_pages.json"
    return output_dir / f"{bounded_intermediate_pdf_stem(pdf_path, suffix)}{suffix}"


def temporary_sibling_path(path: Path) -> Path:
    """Return a temporary sibling path for atomic file replacement."""
    return path.with_name(f".{path.name}.tmp")


def save_page_image(image: Image.Image, output_path: Path) -> None:
    """Write a PNG to a temporary path, then replace its destination."""
    temporary = temporary_sibling_path(output_path)
    try:
        image.save(
            temporary,
            format="PNG",
            compress_level=PNG_COMPRESSION_LEVEL,
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Write formatted UTF-8 JSON through a temporary file for atomic replacement."""
    temporary = temporary_sibling_path(output_path)
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def dpi_to_scale(dpi: int) -> float:
    """Convert DPI to PDFium image scale (pixels per 1/72-inch PDF unit)."""
    return validate_dpi(dpi) / 72.0


def render_page_to_image(
    page: pdfium.PdfPage,
    scale: float,
) -> tuple[pdfium.PdfBitmap, Image.Image]:
    """Render a PDF page and return its bitmap and Pillow image view."""
    bitmap = page.render(
        scale=scale,
        rotation=0,
        grayscale=RENDER_GRAYSCALE,
        fill_color=(255, 255, 255, 255),
        draw_annots=DRAW_ANNOTATIONS,
    )
    return bitmap, bitmap.to_pil()


def render_one_page(
    pdf_document: pdfium.PdfDocument,
    pdf_path: Path,
    output_dir: Path,
    page_index: int,
    dpi: int,
    scale: float,
) -> dict[str, Any]:
    """Render and save one indexed PDF page; release bitmap resources.

    Return image location, dimensions and render diagnostics.
    """
    page_number = page_index + 1
    output_path = make_page_image_path(pdf_path, output_dir, page_number)
    started = time.perf_counter()
    page = pdf_document[page_index]
    bitmap = None
    image = None

    try:
        bitmap, image = render_page_to_image(page, scale)
        width, height = image.size
        save_page_image(image, output_path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to render page {page_number} of {pdf_path.name}"
        ) from error
    # Release the Pillow view before the PDFium bitmap it may reference.
    finally:
        # Pillow may reference the PDFium bitmap, so release in this order.
        if image is not None:
            image.close()
        if bitmap is not None:
            bitmap.close()
        page.close()

    return {
        "page_number": page_number,
        "image_path": str(output_path),
        "width_pixels": int(width),
        "height_pixels": int(height),
        "dpi": dpi,
        "image_mode": "L" if RENDER_GRAYSCALE else "RGB",
        "render_seconds": round(time.perf_counter() - started, 6),
    }


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    """Render every page at the supplied DPI and return a manifest.

    Release the PDF document even when processing fails.
    """
    dpi = validate_dpi(dpi)
    scale = dpi_to_scale(dpi)
    started = time.perf_counter()
    document = pdfium.PdfDocument(str(pdf_path))
    pages: list[dict[str, Any]] = []

    try:
        page_count = len(document)
        if page_count <= 0:
            raise ValueError(f"PDF contains no pages: {pdf_path}")

        for page_index in range(page_count):
            record = render_one_page(
                document,
                pdf_path,
                output_dir,
                page_index,
                dpi,
                scale,
            )
            pages.append(record)
            print(
                f"Rendered page {record['page_number']}/{page_count}: "
                f"{record['image_path']} ({record['render_seconds']:.3f}s)",
                flush=True,
            )
    finally:
        document.close()

    return {
        "source_pdf": str(pdf_path.resolve()),
        "output_dir": str(output_dir),
        "page_count": len(pages),
        "dpi": dpi,
        "grayscale": RENDER_GRAYSCALE,
        "png_compression_level": PNG_COMPRESSION_LEVEL,
        "total_render_seconds": round(time.perf_counter() - started, 6),
        "pages": pages,
    }


def main() -> None:
    """Parse PDF-rendering CLI options, save images and write the JSON manifest."""
    parser = argparse.ArgumentParser(
        description="Render every PDF page as a lossless grayscale PNG."
    )
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for rendered pages. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render resolution in DPI. Default: {DEFAULT_DPI}",
    )
    args = parser.parse_args()

    pdf_path = validate_pdf_path(Path(args.pdf))
    output_dir = ensure_output_dir(Path(args.out_dir))
    manifest = render_pdf_pages(pdf_path, output_dir, validate_dpi(args.dpi))
    manifest_path = make_manifest_path(pdf_path, output_dir)
    save_json(manifest, manifest_path)

    print()
    print(f"Rendered pages: {manifest['page_count']}")
    print(f"Total render time: {manifest['total_render_seconds']:.3f}s")
    print(f"Page images: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
