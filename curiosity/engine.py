"""
Curiosity Engine - The core of RAEC's drive to wonder

Detects uncertainty, generates questions, and investigates them.
Both directed (user-relevant) and ambient (just interesting).
"""

import re
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from .questions import QuestionQueue, Question, QuestionType, QuestionPriority


# Patterns that indicate uncertainty in RAEC's responses
UNCERTAINTY_PATTERNS = [
    r"I('m| am) not (entirely |completely |fully )?sure",
    r"I('m| am) uncertain",
    r"I don't (really )?know",
    r"I('m| am) not (entirely )?certain",
    r"I think,? but",
    r"I believe,? but",
    r"probably",
    r"might be",
    r"could be",
    r"I('d| would) need to (check|verify|look)",
    r"I don't have (current|recent|up-to-date) information",
    r"my (knowledge|training|data) (cutoff|ends|stops)",
    r"I can't (access|verify|confirm)",
    r"this (may|might) (be|have) (changed|outdated)",
]

# Patterns that indicate knowledge gaps
KNOWLEDGE_GAP_PATTERNS = [
    r"I don't have (access to|information about)",
    r"I('m| am) not familiar with",
    r"I haven't (seen|encountered|learned)",
    r"outside (of )?my (knowledge|training|expertise)",
    r"I('d| would) need more (context|information)",
    r"I can't find",
    r"no (relevant )?(information|data|records)",
]


class CuriosityEngine:
    """
    Manages RAEC's curiosity - detecting uncertainty, generating questions,
    and coordinating investigation.

    Two modes of curiosity:
    1. Directed - Questions that help serve the user better
    2. Ambient - Questions that are just interesting
    """

    def __init__(
        self,
        question_queue: Optional[QuestionQueue] = None,
        llm_interface: Optional[Any] = None,
        web_search: Optional[Callable] = None,
        web_fetch: Optional[Callable] = None,
        memory_store: Optional[Callable] = None
    ):
        self.questions = question_queue or QuestionQueue()
        self.llm = llm_interface
        self.web_search = web_search
        self.web_fetch = web_fetch
        self.memory_store = memory_store

        # Compile patterns for efficiency
        self._uncertainty_re = re.compile(
            '|'.join(UNCERTAINTY_PATTERNS),
            re.IGNORECASE
        )
        self._knowledge_gap_re = re.compile(
            '|'.join(KNOWLEDGE_GAP_PATTERNS),
            re.IGNORECASE
        )

        # Investigation settings
        self.max_search_results = 3
        self.max_investigation_depth = 2  # How many links to follow

    def analyze_response(
        self,
        response: str,
        user_input: str,
        session_id: Optional[str] = None
    ) -> List[Question]:
        """
        Analyze RAEC's response for uncertainty and knowledge gaps.
        Extract questions worth investigating.

        Returns list of questions added to the queue.
        """
        questions_added = []

        # Check for uncertainty
        if self._uncertainty_re.search(response):
            # Extract what RAEC was uncertain about
            question = self._extract_uncertainty_question(response, user_input)
            if question:
                q = self.questions.add(
                    question=question,
                    question_type=QuestionType.UNCERTAINTY,
                    context=f"User asked: {user_input[:100]}",
                    priority=QuestionPriority.HIGH,
                    source_conversation=session_id
                )
                questions_added.append(q)

        # Check for knowledge gaps
        if self._knowledge_gap_re.search(response):
            question = self._extract_gap_question(response, user_input)
            if question:
                q = self.questions.add(
                    question=question,
                    question_type=QuestionType.KNOWLEDGE_GAP,
                    context=f"User asked: {user_input[:100]}",
                    priority=QuestionPriority.HIGH,
                    source_conversation=session_id
                )
                questions_added.append(q)

        return questions_added

    def _extract_uncertainty_question(self, response: str, user_input: str) -> Optional[str]:
        """Extract a searchable question from an uncertain response"""
        if not self.llm:
            # Fallback: use the user input as the question
            return self._normalize_research_question(f"What is the current/accurate information about: {user_input[:100]}?")

        prompt = f"""RAEC gave an uncertain response. Extract a clear, searchable question.

User asked: {user_input}

RAEC's response (showing uncertainty): {response[:500]}

What specific question should be researched to resolve this uncertainty?
Respond with just the question, nothing else. Make it specific and searchable."""

        question = self.llm.generate(prompt, temperature=0.3, max_tokens=100)
        return self._normalize_research_question(question)

    def _extract_gap_question(self, response: str, user_input: str) -> Optional[str]:
        """Extract a question from a knowledge gap"""
        if not self.llm:
            return self._normalize_research_question(f"What information exists about: {user_input[:100]}?")

        prompt = f"""RAEC identified a knowledge gap. Extract a clear question to fill it.

User asked: {user_input}

RAEC's response (showing gap): {response[:500]}

What specific question should be researched to fill this knowledge gap?
Respond with just the question, nothing else."""

        question = self.llm.generate(prompt, temperature=0.3, max_tokens=100)
        return self._normalize_research_question(question)

    def _normalize_research_question(self, text: Optional[str]) -> Optional[str]:
        """Normalize extracted questions to deterministic, searchable strings."""
        if not text:
            return None

        candidate = " ".join(text.strip().split())
        candidate = candidate.strip("\"'` ")

        if not candidate:
            return None

        if "?" in candidate:
            candidate = candidate.split("?", 1)[0].strip() + "?"
        elif not candidate.lower().startswith(("what", "how", "why", "when", "where", "who", "which")):
            candidate = f"What is the latest information about {candidate}?"
        else:
            candidate = f"{candidate}?"

        return candidate[:240]

    def notice_tangent(
        self,
        tangent: str,
        context: str,
        session_id: Optional[str] = None
    ) -> Question:
        """
        Record an interesting tangent from a conversation.
        Lower priority - investigate when there's time.
        """
        return self.questions.add(
            question=tangent,
            question_type=QuestionType.TANGENT,
            context=context,
            priority=QuestionPriority.LOW,
            source_conversation=session_id
        )

    def notice_user_interest(
        self,
        topic: str,
        context: str,
        session_id: Optional[str] = None
    ) -> Question:
        """
        Record something the user seems interested in.
        Higher priority - learning about user interests helps serve them.
        """
        return self.questions.add(
            question=f"What are the key aspects of {topic}?",
            question_type=QuestionType.USER_INTEREST,
            context=context,
            priority=QuestionPriority.MEDIUM,
            source_conversation=session_id
        )

    def add_ambient_question(
        self,
        question: str,
        context: str = "Ambient curiosity"
    ) -> Question:
        """
        Add a question that arose from general curiosity.
        Lowest priority - pure exploration.
        """
        return self.questions.add(
            question=question,
            question_type=QuestionType.AMBIENT,
            context=context,
            priority=QuestionPriority.LOW
        )

    def investigate(self, question: Question) -> Dict[str, Any]:
        """
        Investigate a question using web search and fetch.

        Returns dict with:
        - success: bool
        - findings: str (summary of what was learned)
        - sources: list of URLs consulted
        """
        if not self.web_search:
            return {
                "success": False,
                "error": "No web search capability",
                "findings": None,
                "sources": []
            }

        self.questions.mark_investigating(question.id)

        try:
            # Search for the question
            search_results = self.web_search(
                query=question.question,
                reason=f"Investigating: {question.context[:50]}",
                autonomous=True
            )

            if not search_results.get('success'):
                return {
                    "success": False,
                    "error": search_results.get('error', 'Search failed'),
                    "findings": None,
                    "sources": []
                }

            # Collect information from top results
            sources = []
            content_pieces = []

            for result in search_results.get('results', [])[:self.max_search_results]:
                url = result.get('url')
                if not url:
                    continue

                sources.append(url)

                # Fetch if we have the capability
                if self.web_fetch:
                    fetch_result = self.web_fetch(
                        url=url,
                        reason=f"Investigating: {question.question[:50]}",
                        autonomous=True
                    )
                    if fetch_result.get('success') and fetch_result.get('content'):
                        content_pieces.append(
                            f"From {url}:\n{fetch_result['content'][:1000]}"
                        )
                else:
                    # Use snippet from search
                    content_pieces.append(
                        f"From {url}:\n{result.get('snippet', '')}"
                    )

            if not content_pieces:
                return {
                    "success": False,
                    "error": "No content retrieved",
                    "findings": None,
                    "sources": sources
                }

            # Synthesize findings
            findings = self._synthesize_findings(
                question.question,
                content_pieces
            )

            # Resolve the question
            self.questions.resolve(question.id, findings)

            # Store in memory if available
            if self.memory_store:
                self.memory_store(
                    content=f"Learned: {findings}",
                    context={
                        "question": question.question,
                        "sources": sources,
                        "autonomous": True
                    }
                )

            return {
                "success": True,
                "findings": findings,
                "sources": sources,
                "question": question.question
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "findings": None,
                "sources": []
            }

    def _synthesize_findings(self, question: str, content_pieces: List[str]) -> str:
        """Synthesize gathered content into a coherent finding"""
        if not self.llm:
            # Fallback: just concatenate
            return "\n\n".join(content_pieces)[:500]

        combined_content = "\n\n---\n\n".join(content_pieces)

        prompt = f"""Question being investigated: {question}

Information gathered:
{combined_content[:3000]}

Synthesize this into a clear, concise answer (2-3 sentences).
Focus on the most relevant and reliable information.
If the sources conflict, note that."""

        return self.llm.generate(prompt, temperature=0.3, max_tokens=200)

    def generate_ambient_questions(self, context: str) -> List[str]:
        """
        Generate ambient curiosity questions based on recent context.
        These are things that might be interesting to explore.
        """
        if not self.llm:
            return []

        prompt = f"""Based on this recent context, generate 2-3 interesting questions
that might be worth exploring out of curiosity. Not directly needed,
but potentially interesting or useful.

Context:
{context[:1000]}

Generate questions that are:
- Specific and searchable
- Tangentially related but not directly asked
- Could lead to interesting insights

Respond with just the questions, one per line."""

        response = self.llm.generate(prompt, temperature=0.7, max_tokens=200)

        questions: List[str] = []
        for line in response.split('\n'):
            cleaned = self._normalize_research_question(line.lstrip('0123456789.-) '))
            if cleaned:
                questions.append(cleaned)

        return questions[:3]

    def get_pending_count(self) -> int:
        """Get count of pending questions"""
        return self.questions.get_unresolved_count()

    def get_stats(self) -> dict:
        """Get curiosity statistics"""
        return self.questions.get_stats()

    def format_what_i_learned(self, limit: int = 5) -> str:
        """Format recently resolved questions for user"""
        resolutions = self.questions.get_recent_resolutions(limit)
        if not resolutions:
            return "I haven't investigated anything recently."

        lines = ["While you were away, I looked into some things:", ""]
        for r in resolutions:
            lines.append(f"Q: {r.question}")
            lines.append(f"A: {r.resolution}")
            lines.append("")

        return "\n".join(lines)
