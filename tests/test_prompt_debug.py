import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observability.prompt_debug import PromptDebugLogger


def test_prompt_debug_logger_writes_json(tmp_path: Path):
    logger = PromptDebugLogger(enabled=True, export_dir=str(tmp_path), print_prompt=False)

    output_path = logger.log(
        source="chat",
        prompt="User: hi",
        metadata={"task_type": "synthesis", "components": {"history_count": 2}},
    )

    assert output_path is not None
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert payload["source"] == "chat"
    assert payload["prompt"] == "User: hi"
    assert payload["metadata"]["task_type"] == "synthesis"


def test_prompt_debug_logger_noop_when_disabled(tmp_path: Path):
    logger = PromptDebugLogger(enabled=False, export_dir=str(tmp_path), print_prompt=False)
    output_path = logger.log(source="query", prompt="Question", metadata={})
    assert output_path is None
    assert list(tmp_path.iterdir()) == []
