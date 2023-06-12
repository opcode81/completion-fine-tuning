import logging
import sys

from completionft.completion_task import CompletionTaskModelComparison

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    ruby_models = [
        "checkpoints/ruby/checkpoint-6000",
        "checkpoints/ruby/checkpoint-500",
        "bigcode/santacoder"
    ]
    CompletionTaskModelComparison("ruby", models=ruby_models).run()
    #CompletionTaskModelComparison("c-sharp", models=["bigcode/santacoder"]).run()
    log.info("Done")
