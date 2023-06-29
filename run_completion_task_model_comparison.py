import logging
import sys

from completionft.completion_model_comparison import CompletionTaskModelComparison
from completionft.model import SantaCoderModelFactory

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    model_factory = SantaCoderModelFactory()

    general_models = ["bigcode/santacoder"]
    ruby_models = general_models + [
        "checkpoints/ruby/checkpoint-6000",
        "checkpoints/ruby/checkpoint-500"
    ]
    c_sharp_models = general_models + [
        "checkpoints/c-sharp/checkpoint-1000",
        "checkpoints/c-sharp/checkpoint-2000",
        "checkpoints/c-sharp/checkpoint-3000",
        "checkpoints/c-sharp/checkpoint-4000"
    ]
    rust_models = general_models + [
        "checkpoints/rust/checkpoint-30000"
    ]

    #CompletionTaskModelComparison("ruby", model_factory, model_paths=ruby_models).run()
    #CompletionTaskModelComparison("c-sharp", model_factory, model_paths=c_sharp_models).run()
    CompletionTaskModelComparison("rust", model_factory, model_paths=rust_models).run()
    #CompletionTaskModelComparison("scala", model_factory, model_paths=general_models).run()

    log.info("Done")
