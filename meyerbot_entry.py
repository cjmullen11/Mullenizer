"""Unified entry point for source and PyInstaller-frozen."""

from __future__ import annotations

import sys
from multiprocessing import freeze_support


PIPELINE_FLAG = "--meyerbot-pipeline"
STAGE_FLAG = "--meyerbot-stage"

VALID_STAGES = ("render", "deskew", "ocr")


def call_module_main(module_name: str, arguments: list[str]) -> None:
    """Dispatch a stage or GUI module using temporary command-line arguments.

    Explicit imports let PyInstaller discover the modules without hidden imports.
    """
    if module_name == "meyerbot_gui":
        import meyerbot_gui as module
    elif module_name == "run_detection_pipeline":
        import run_detection_pipeline as module
    elif module_name == "render_pdf":
        import render_pdf as module
    elif module_name == "deskew_pages":
        import deskew_pages as module
    elif module_name == "ocr_report_targets":
        import ocr_report_targets as module
    else:
        raise RuntimeError(f"Unknown module: {module_name}")

    module_main = getattr(module, "main", None)

    if module_main is None:
        raise RuntimeError(f"Module has no main() function: {module_name}")

    original_argv = sys.argv
    sys.argv = [f"{module_name}.py", *arguments]

    try:
        module_main()
    finally:
        sys.argv = original_argv


def main() -> None:
    """Dispatch the GUI, batch pipeline or an internal stage in source/frozen mode.

    Initialize multiprocessing support before starting workers.
    """
    # Required before ProcessPoolExecutor starts child processes from a frozen
    # Windows executable. It is harmless during ordinary source execution.
    freeze_support()

    arguments = sys.argv[1:]

    if arguments and arguments[0] == PIPELINE_FLAG:
        call_module_main("run_detection_pipeline", arguments[1:])
        return

    if arguments and arguments[0] == STAGE_FLAG:
        if len(arguments) < 2:
            raise SystemExit(f"{STAGE_FLAG} requires a stage name.")

        stage_name = arguments[1]

        if stage_name == "render":
            module_name = "render_pdf"
        elif stage_name == "deskew":
            module_name = "deskew_pages"
        elif stage_name == "ocr":
            module_name = "ocr_report_targets"
        else:
            choices = ", ".join(VALID_STAGES)
            raise SystemExit(
                f"Unknown stage {stage_name!r}. Choices: {choices}"
            )

        call_module_main(module_name, arguments[2:])
        return

    call_module_main("meyerbot_gui", arguments)


if __name__ == "__main__":
    main()
