from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curiosity.engine import CuriosityEngine


def test_normalize_research_question_adds_terminal_question_mark() -> None:
    engine = CuriosityEngine(llm_interface=None)
    normalized = engine._normalize_research_question("How does ROCm scheduler tuning work")
    assert normalized == "How does ROCm scheduler tuning work?"


def test_normalize_research_question_limits_multiline_and_quotes() -> None:
    engine = CuriosityEngine(llm_interface=None)
    normalized = engine._normalize_research_question('"1) What changed in ollama model routing?\nWith examples"')
    assert normalized == "1) What changed in ollama model routing?"


def test_extract_uncertainty_question_uses_normalized_fallback() -> None:
    engine = CuriosityEngine(llm_interface=None)
    question = engine._extract_uncertainty_question(
        response="I'm not sure.",
        user_input="rocm memory pressure behavior",
    )
    assert question is not None
    assert question.endswith("?")
