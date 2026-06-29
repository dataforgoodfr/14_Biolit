"""Worker persistant exécutant périodiquement le cycle BioLit."""

import os
import signal
import threading

import structlog

from biolit.settings import PIPELINE_INTERVAL_SECONDS
from pipelines.run import Runtime, run_cycle

LOGGER = structlog.get_logger()


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    runtime = Runtime.from_environment()
    run_once = os.getenv("PIPELINE_RUN_ONCE", "false").lower() == "true"
    while not stop.is_set():
        try:
            run_cycle(runtime)
        except Exception:
            LOGGER.exception("pipeline_cycle_failed")
            if run_once:
                raise
        if run_once:
            break
        stop.wait(PIPELINE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
