"""Module entry point for ``python -m llm_eval``."""

import sys

from llm_eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
