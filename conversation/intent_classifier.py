"""
Intent Classifier for RAEC

Classifies user input into interaction modes:
- CHAT: Casual conversation, greetings, small talk
- QUERY: Information requests, questions about facts/concepts
- TASK: Action requests requiring tool use or code execution
- META: Commands about RAEC itself (settings, status, help)
"""
import re
from enum import Enum
from typing import Tuple, List, Optional
from dataclasses import dataclass


class Intent(Enum):
    CHAT = "chat"       # Casual conversation
    QUERY = "query"     # Information request
    TASK = "task"       # Action/execution request
    META = "meta"       # System commands about RAEC


@dataclass
class ClassificationResult:
    """Result of intent classification"""
    intent: Intent
    confidence: float  # 0.0 to 1.0
    keywords: List[str]  # Keywords that influenced classification
    reasoning: str  # Brief explanation


class IntentClassifier:
    """
    Rule-based intent classifier with pattern matching.

    Uses keyword patterns and structural analysis to determine
    the most likely intent without requiring LLM calls.
    """

    # Patterns for each intent type
    TASK_PATTERNS = [
        # Action verbs
        r'\b(create|make|build|write|generate|delete|remove|move|copy|rename)\b',
        r'\b(run|execute|install|download|upload|save|open|close)\b',
        r'\b(add|update|modify|change|edit|fix|set|configure)\b',
        r'\b(send|post|submit|deploy|publish|start|stop|restart)\b',
        # File operations
        r'\b(file|folder|directory|path)\b.*\b(create|make|delete|move)\b',
        # Explicit task language
        r'\b(can you|could you|please|would you)\b.*\b(create|make|do|help)\b',
        r'\b(i need|i want)\b.*\b(to|you to)\b',
    ]

    QUERY_PATTERNS = [
        # Question words
        r'^(what|who|where|when|why|how|which|whose)\b',
        r'\b(what is|what are|what does|what do)\b',
        r'\b(how do|how does|how can|how to)\b',
        r'\b(explain|describe|tell me about|define)\b',
        # Information seeking
        r'\b(meaning|definition|difference between)\b',
        r'\?$',  # Ends with question mark
    ]

    CHAT_PATTERNS = [
        # Greetings
        r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
        r'^(sup|yo|howdy|hiya)\b',
        # Farewells
        r'\b(bye|goodbye|see you|later|farewell|take care)\b',
        # Social
        r'\b(thanks|thank you|appreciate|grateful)\b',
        r'^(how are you|how\'s it going|what\'s up)\b',
        # Casual affirmations
        r'^(ok|okay|sure|cool|nice|great|awesome|got it|understood)\b',
        r'^(yes|no|yeah|nah|yep|nope)\b',
    ]

    META_PATTERNS = [
        # RAEC-specific commands
        r'^/(help|status|settings|config|reset|clear|skills|identity)\b',
        r'\b(your (name|purpose|capabilities|skills|version))\b',
        r'\b(raec|this (system|agent|ai))\b',
        r'\b(who are you|what are you|what can you do)\b',
        # Settings/configuration
        r'\b(change your|update your|modify your)\b',
        r'\b(your (settings|preferences|behavior))\b',
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.task_patterns = [re.compile(p, re.IGNORECASE) for p in self.TASK_PATTERNS]
        self.query_patterns = [re.compile(p, re.IGNORECASE) for p in self.QUERY_PATTERNS]
        self.chat_patterns = [re.compile(p, re.IGNORECASE) for p in self.CHAT_PATTERNS]
        self.meta_patterns = [re.compile(p, re.IGNORECASE) for p in self.META_PATTERNS]

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify user input into an intent category.

        Returns ClassificationResult with intent, confidence, and reasoning.
        """
        text = text.strip()

        # Check for empty input
        if not text:
            return ClassificationResult(
                intent=Intent.CHAT,
                confidence=1.0,
                keywords=[],
                reasoning="Empty input treated as chat"
            )

        # Score each intent
        scores = {
            Intent.TASK: self._score_patterns(text, self.task_patterns),
            Intent.QUERY: self._score_patterns(text, self.query_patterns),
            Intent.CHAT: self._score_patterns(text, self.chat_patterns),
            Intent.META: self._score_patterns(text, self.meta_patterns),
        }

        # Get best match
        best_intent = max(scores, key=lambda k: scores[k][0])
        best_score, keywords = scores[best_intent]

        # Calculate confidence (normalize against total)
        total_score = sum(s[0] for s in scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            # No patterns matched - default to CHAT with low confidence
            confidence = 0.3
            best_intent = Intent.CHAT

        # Apply heuristics for edge cases
        best_intent, confidence = self._apply_heuristics(text, best_intent, confidence, scores)

        return ClassificationResult(
            intent=best_intent,
            confidence=round(confidence, 2),
            keywords=keywords,
            reasoning=self._generate_reasoning(best_intent, keywords, confidence)
        )

    def _score_patterns(self, text: str, patterns: List[re.Pattern]) -> Tuple[float, List[str]]:
        """Score text against a list of patterns"""
        score = 0.0
        matched_keywords = []

        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                score += len(matches)
                # Extract matched text
                for match in matches:
                    if isinstance(match, tuple):
                        matched_keywords.extend(match)
                    else:
                        matched_keywords.append(match)

        return score, list(set(matched_keywords))

    def _apply_heuristics(self, text: str, intent: Intent, confidence: float,
                          scores: dict) -> Tuple[Intent, float]:
        """Apply additional heuristics for edge cases"""

        # Short inputs (< 4 words) with no strong signals default to CHAT
        word_count = len(text.split())
        if word_count < 4 and confidence < 0.5:
            return Intent.CHAT, 0.4

        # Long inputs with action verbs are likely TASK even if query-like
        if word_count > 10 and scores[Intent.TASK][0] > 0:
            if scores[Intent.TASK][0] >= scores[Intent.QUERY][0]:
                return Intent.TASK, min(confidence + 0.1, 1.0)

        # Commands starting with "/" are always META
        if text.startswith('/'):
            return Intent.META, 1.0

        # File paths or code snippets suggest TASK
        if re.search(r'[/\\][\w\.]+|```|`[^`]+`', text):
            if intent not in [Intent.TASK, Intent.META]:
                return Intent.TASK, max(confidence, 0.6)

        return intent, confidence

    def _generate_reasoning(self, intent: Intent, keywords: List[str], confidence: float) -> str:
        """Generate brief reasoning for classification"""
        if not keywords:
            return f"Defaulted to {intent.value} (no strong signals)"

        keyword_str = ", ".join(keywords[:3])
        if confidence >= 0.7:
            return f"Strong {intent.value} signals: {keyword_str}"
        elif confidence >= 0.4:
            return f"Moderate {intent.value} indicators: {keyword_str}"
        else:
            return f"Weak {intent.value} match: {keyword_str}"

    def get_routing_suggestion(self, result: ClassificationResult) -> str:
        """
        Get a routing suggestion based on classification.

        Returns guidance on how RAEC should handle this input.
        """
        intent = result.intent

        if intent == Intent.CHAT:
            return "respond_conversationally"
        elif intent == Intent.QUERY:
            return "provide_information"
        elif intent == Intent.TASK:
            return "create_and_execute_plan"
        elif intent == Intent.META:
            return "handle_system_command"

        return "respond_conversationally"  # Fallback


# Convenience function
def classify_intent(text: str) -> ClassificationResult:
    """Quick intent classification"""
    classifier = IntentClassifier()
    return classifier.classify(text)


if __name__ == "__main__":
    # Test examples
    classifier = IntentClassifier()

    test_inputs = [
        "Hello!",
        "Create a file called test.txt on my desktop",
        "What is machine learning?",
        "/help",
        "Thanks for your help!",
        "Can you explain how Python decorators work?",
        "Run the tests and fix any failures",
        "Who are you?",
        "What can you do?",
        "I need you to write a script that downloads images",
    ]

    print("Intent Classification Test\n" + "=" * 50)
    for text in test_inputs:
        result = classifier.classify(text)
        print(f"\nInput: {text}")
        print(f"  Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")
