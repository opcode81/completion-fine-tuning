import logging
import sys

from completionft.eval import ModelPerplexityEvaluation

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    ruby_models = [
        "checkpoints/ruby-lora16-fp32/checkpoint-3500",
        "checkpoints/ruby-lora64-fp32/checkpoint-3000",
        "checkpoints/ruby/checkpoint-3000",
        "checkpoints/ruby/checkpoint-500",
        "checkpoints/ruby/checkpoint-1000",
        "checkpoints/ruby/checkpoint-2000",
        "bigcode/santacoder"
    ]
    ModelPerplexityEvaluation("ruby", ruby_models).run()