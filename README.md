# Mullenizer

**Status:** Testing / Finalizing  
**Technologies:** Python · OpenCV · Tesseract OCR · NumPy · Tkinter · PDF processing

## Overview

Mullenizer is a Python desktop application for extracting and reviewing information from cardiac device reports in PDF format. It uses primarily OCR-based document processing to identify report data and combines automated detection with human-guided interpretation of graphs. The goal is to streamline data collection for report reviewers, not replace their judgment or remove human oversight.

This project is not validated for any specific study or other use. It serves as a proof of concept.

The name *Mullenizer* is derived from my surname.

**All extracted data must be independently validated.** No claims are made regarding the application's privacy, reliability, or accuracy.

## Key Design Choices

- **OCR-based extraction:** Render, deskew, and analyze PDF pages to identify demographics, atrial fibrillation (AFib) burden, and heart-rate and activity data.
- **Human-guided review:** Highlight likely matches for reviewers to verify rather than treating detected results as definitive.
- **Graph digitization:** Let reviewers manually select and interpret line graphs and histograms.
- **Desktop-first design:** Use a lightweight Tkinter interface, chosen over a web app because of privacy-related constraints, ease of use, and available resources. This does not guarantee data privacy.
- **Structured export:** Save reviewed data to CSV or JSON.

## Project Structure

| File | Purpose |
|---|---|
| `mullenizer_entry.py` | Application entry point |
| `mullenizer_gui.py` | Desktop interface |
| `run_detection_pipeline.py` | Processing orchestration |
| `render_pdf.py` | PDF rendering |
| `deskew_pages.py` | Page alignment |
| `ocr_report_targets.py` | OCR and information extraction |
| `human_guided_graph_interpreter.py` | Interactive graph digitization |

## Roadmap

- [ ] Complete testing and finalize the desktop release.
- [ ] Improve documentation and refine the implementation.
- [ ] Add GUI controls for adjusting OCR keywords and target metrics.
- [ ] Support reports that do not require deskewing.
- [ ] Evaluate a web-based interface as a possible future extension.

## Usage

### Requirements

**For source-based development:** Python 3, `opencv-python`, `numpy`, `Pillow`, `pypdfium2`, `pytesseract`, Tkinter, and Tesseract OCR with English language data.

### Setup

On Windows, download and run the installer from the repository’s **Releases** page. Launch Mullenizer like any other installed application.

### Configuration

Additional configuration instructions are not currently documented.

### Operation

Open a cardiac device report, verify the extracted fields, manually interpret relevant graphs, and export reviewed results to CSV or JSON. **Check all outputs against the original report.**

## Acknowledgments

Thanks to the OpenCV and Tesseract teams, the maintainers of other dependencies, and the open-source community for sharing tools and examples that made this project possible.

This was my first project using several of these libraries. The code and documentation can be improved, and I’m happy to clarify implementation choices for anyone extending the project.
