# Mullenizer

Mullenizer is a Python desktop application for extracting and reviewing information from cardiac device reports (PDFs). It combines automated document processing with human-guided graph interpretation. It is primarily OCR based. Its goal is to streamline data collection by acting as a tool to report reviewers, NOT to replace reviewers or provide automation. All data should be validated; no claims are made about privacy, reliability, or accuracy.

Thank you to the teams behind OpenCV, tesseract, all other dependencies, and the open source community as a whole - much of this was made very easy by the generous sharing of other similar projects. 

## Running
 On linux, you can run meyerbot_gui.py directly as described below. Ensure all dependencies are available.
 Install the Python dependencies and Tesseract OCR, then launch:

```bash
python meyerbot_entry.py
```

 On windows, use the published installer. Run as any normal app.

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

In testing / finalizing.

Possible extensions/revisions:

 - Web based version
      This project is setup well to be converted into this. Due to the constraints it was created under (privacy, ease of use, and resources), it was determined that a "small" local gui would be more appropriate.

- GUI for self adjustment of keywords and metrics to be searched for

- Much optimization could be done, in terms of documentation and functionality/code.

- A deskew-less version for those that it is not applicable to.



I am happy to clarify anything to assist with the extensions of others, however do consider this was my first project using many of these libraries.
