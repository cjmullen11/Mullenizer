# Mullenizer 1.0

Mullenizer is a Python desktop application for extracting and reviewing information from cardiac device reports (PDFs). It combines automated document processing with human-guided graph interpretation.

## Features

- **PDF processing:** Render, deskew, and OCR report pages.
- **Data extraction:** Identify patient demographics, AFib burden, and relevant heart rate/activity information.
- **Guided review:** Highlight likely matches for human verification.
- **Graph interpretation:** Manually select and digitize line graphs and histograms.
- **Export:** Save reviewed measurements and results to CSV/JSON.

## Dependencies

- **Python 3** with `opencv-python`, `numpy`, `Pillow`, `pypdfium2`, and `pytesseract`
- **Tkinter** for the desktop GUI
- **Tesseract OCR** with English language data

## Running

Install the Python dependencies and Tesseract OCR, then launch:

```bash
python mullenizer_entry.py
```

Packaged Windows releases can include all dependencies.

## Project Structure

| File | Purpose |
|---|---|
| `mullenizer_entry.py` | Application entry point |
| `mullenizer_gui.py` | Desktop interface |
| `run_detection_pipeline.py` | Processing orchestration |
| `render_pdf.py` | PDF rendering |
| `deskew_pages.py` | Image alignment |
| `ocr_report_targets.py` | OCR and information extraction |
| `human_guided_graph_interpreter.py` | Graph digitization |

## Status

Under active development. Automated extraction and graph interpretation require human review.
