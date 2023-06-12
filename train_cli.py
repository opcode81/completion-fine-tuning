"""
Command line interface for fine-tuning
"""

import logging
import sys

import jsonargparse

from completionft.finetuning import FineTuningConfiguration, CompletionFineTuning

log = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    cfg = jsonargparse.CLI(FineTuningConfiguration, as_positional=False)
    CompletionFineTuning(cfg).run()
