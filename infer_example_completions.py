import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
import re

import torch
from transformers import pipeline, AutoTokenizer

log = logging.getLogger(__name__)


TAG_COMPLETION_PLACEHOLDER = "<todo>"
TAG_FIM_PREFIX = "<fim-prefix>"
TAG_FIM_SUFFIX = "<fim-suffix>"
TAG_FIM_MIDDLE = "<fim-middle>"

re_todo_tag = re.compile(re.escape(TAG_COMPLETION_PLACEHOLDER))
re_fim_middle = re.compile(re.escape(TAG_FIM_MIDDLE))


class CompletionTask:
    def __init__(self, prefix: str, suffix: str, lang_id: str):
        self.prefix = prefix
        self.suffix = suffix
        self.middle = None
        self.lang_id = lang_id

    @classmethod
    def from_code_with_todo_tag(cls, code_with_todo: str, lang_id: str) -> "CompletionTask":
        m = re_todo_tag.search(code_with_todo)
        assert m is not None
        prefix = code_with_todo[:m.start()]
        suffix = code_with_todo[m.end():]
        return cls(prefix, suffix, lang_id)

    def fim_prompt(self) -> str:
        return f"{TAG_FIM_PREFIX}{self.prefix}{TAG_FIM_SUFFIX}{self.suffix}{TAG_FIM_MIDDLE}"

    def _extract_completion(self, s: str) -> str:
        m = re_fim_middle.search(s)
        if not m:
            return ""
        completion = s[m.end():]

        # truncate completion depending on lang_id
        if self.lang_id == "ruby":  # end completion after first occurrence of 'end'
            m = re.search(r"end\n", completion)
            if m:
                completion = completion[:m.end()]
                if self.suffix.startswith("\n"):
                    completion = completion[:-1]

        return completion

    def apply_completion(self, model_response: str) -> None:
        self.middle = self._extract_completion(model_response)

    def full_code(self) -> str:
        return self.prefix + self.middle + self.suffix


def read_completion_tasks(lang_id) -> Dict[str, CompletionTask]:
    tasks = {}
    root = Path("data") / "completion-tasks" / lang_id
    for fn in os.listdir(root):
        with open(root / fn, "r") as f:
            code = f.read()
            tasks[fn] = CompletionTask.from_code_with_todo_tag(code, lang_id)
    return tasks


def run(models: List[str], lang_id: str, device="cuda:0", base_model_id="bigcode/santacoder"):
    tasks = read_completion_tasks(lang_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    root = Path("outputs") / "completion-results" / lang_id
    for model_id in models:

        log.info(f"Loading model {model_id}")
        pipe = pipeline("text-generation", model=model_id, max_new_tokens=256, device=device,
            torch_dtype=torch.bfloat16, trust_remote_code=True, tokenizer=tokenizer)

        model_path = root / model_id

        for task_name, task in tasks.items():
            log.info(f"Querying {model_id} for completion {task_name}")
            prompt = task.fim_prompt()
            response = pipe(prompt)[0]["generated_text"]
            task.apply_completion(response)
            response_code = task.full_code()

            log.info(f"Completion for {task_name} by {model_id}:\n{response_code}")
            task_path = model_path / task_name
            with open(task_path, "w") as f:
                f.write(response_code)
        del pipe


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    run(models=["bigcode/santacoder"], lang_id="c-sharp")
    log.info("Done")
