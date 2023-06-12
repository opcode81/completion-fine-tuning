import logging
import os
import sys
from pathlib import Path

from completionft.completion_task import TAG_COMPLETION_PLACEHOLDER
from completionft.model import model_id_from_fn

log = logging.getLogger(__name__)

HTML_PREFIX = """<html>
<head>
    <style>
        body {
            font-family: sans-serif;
            padding: 30px;
        }
        .code {
            white-space: pre;
            font-family: Courier New, monospace;
            background-color: #eee;
            margin-top: 10px;
            padding: 10px;
        }
        .placeholder {
            color: red;
        }
        .task {
        }
        table.completions {
            width: 100%;
        }
        table.completions .code {
            overflow-x: scroll;
        }
        table.completions td {
            vertical-align: top;
            padding: 10px;
        }
        table.completions td:first-child {
            padding-left: 0;
        }
    </style>
</head>
<body>"""

HTML_SUFFIX = "</body></html>"


def write_html_completions(lang_id: str):
    root_dir = Path("results") / "completion-tasks" / lang_id

    html = HTML_PREFIX
    html += f"<h1>{lang_id}</h1>"

    for task_name in os.listdir(root_dir):
        task_dir = root_dir / task_name
        if not task_dir.is_dir():
            continue

        log.info(f"Task '{task_name}'")
        html += f"<h2>{task_name}</h2>"

        completions = {}
        for fn in os.listdir(task_dir):
            content = (task_dir / fn).read_text()
            basename = os.path.splitext(fn)[0]
            if basename == "task":
                content = content.replace(TAG_COMPLETION_PLACEHOLDER,
                    '<span class="placeholder">&lt;todo&gt;</span>')
                html += f'<h3>Task</h3><div class="code task">{content}</div>'
            else:
                model_name = model_id_from_fn(basename)
                completions[model_name] = content

        html += '<h3>Completions</h3><table class="completions"><tr>'
        columns = 2
        for i, model_name in enumerate(sorted(completions.keys()), start=1):
            completion = completions[model_name]
            html += f'<td style="">{model_name}<br><div class="code">{completion}</td>'
            if i % columns == 0:
                html += "</tr>"
                if len(completions) > i:
                    html += "<tr>"
        html += "</tr></table>"

    html += HTML_SUFFIX

    # write to results file
    result_file_path = root_dir / f"results.html"
    log.info(f"Writing HTML output to {result_file_path}")
    with open(result_file_path, "w") as f:
        f.write(html)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    write_html_completions(lang_id="ruby")
    log.info("Done")