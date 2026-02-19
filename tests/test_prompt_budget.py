from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.prompt_budget import PromptBudgetAnalyzer


def test_prompt_budget_analyzer_reports_lengths_and_threshold() -> None:
    analyzer = PromptBudgetAnalyzer(warn_tokens=10)
    result = analyzer.analyze(
        prompt="x" * 100,
        components={"state_context": "abc", "history_count": 3},
    )

    assert result["prompt_chars"] == 100
    assert result["prompt_tokens_est"] == 25
    assert result["component_chars"]["state_context"] == 3
    assert result["warn_threshold_tokens"] == 10
    assert result["over_budget"] is True
