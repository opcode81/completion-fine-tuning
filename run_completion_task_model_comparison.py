import logging
import sys

from completionft.completion_task import CompletionTaskModelComparison
from completionft.model import SantaCoderModelFactory

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    model_factory = SantaCoderModelFactory()
    ruby_models = [
        "checkpoints/ruby/checkpoint-6000",
        "checkpoints/ruby/checkpoint-500",
        "bigcode/santacoder"
    ]
    CompletionTaskModelComparison("ruby", model_factory, model_paths=ruby_models).run()
    #CompletionTaskModelComparison("c-sharp", model_factory, models=["bigcode/santacoder"]).run()
    log.info("Done")
