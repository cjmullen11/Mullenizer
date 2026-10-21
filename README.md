# Mullenizer

Mullenizer is a Python desktop application for extracting and reviewing information from cardiac device reports (PDFs). It combines automated document processing with human-guided graph interpretation. It is primarily OCR based. Its goal is to streamline data collection by acting as a tool to report reviewers, NOT to replace reviewers or provide automation. All data should be validated. 

Thank you to the teams behind OpenCV, tesseract, all other dependencies, and the open source community as a whole - much of this was made very easy by the generous sharing of other similar projects. 

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
python meyerbot_entry.py
```


## Project Structure

| File | Purpose |
|---|---|
| `meyerbot_entry.py` | Application entry point |
| `meyerbot_gui.py` | Desktop interface |
| `run_detection_pipeline.py` | Processing orchestration |
| `render_pdf.py` | PDF rendering |
| `deskew_pages.py` | Image alignment |
| `ocr_report_targets.py` | OCR and information extraction |
| `human_guided_graph_interpreter.py` | Graph digitization |

## Status

In testing, finalizing.

## License/Use

Feel free to use or fork anything. No claims are made about reliability or accuracy.
