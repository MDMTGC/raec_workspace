"""
Confidence Tracker - Quantifying and calibrating RAEC's uncertainty

This helps RAEC:
- Know when to say "I don't know"
- Calibrate confidence (avoid over/under-confidence)
- Distinguish types of uncertainty
- Improve over time through feedback
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import statistics
import re


class UncertaintyType(Enum):
    """Types of uncertainty"""
    EPISTEMIC = "epistemic"    # Don't know (could learn)
    ALEATORIC = "aleatoric"    # Random/inherent uncertainty
    MODEL = "model"            # LLM limitations
    DATA = "data"              # Missing or stale data


@dataclass
class ConfidenceScore:
    """A confidence assessment"""
    score: float  # 0.0 to 1.0
    uncertainty_type: str
    reasoning: str
    factors: List[str]  # What influenced this score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationStats:
    """Statistics about confidence calibration"""
    total_predictions: int
    avg_confidence: float
    avg_accuracy: float
    calibration_error: float  # Difference between confidence and accuracy
    overconfidence_rate: float
    underconfidence_rate: float
    by_bucket: Dict[str, Dict[str, float]]  # Bucketed calibration


class ConfidenceTracker:
    """
    Tracks and calibrates RAEC's confidence.

    Maintains history of confidence assessments and their outcomes
    to improve calibration over time.
    """

    # Patterns indicating low confidence in text
    LOW_CONFIDENCE_PATTERNS = [
        r"I('m| am) not (entirely |completely |fully )?sure",
        r"I think,? but",
        r"probably",
        r"might be",
        r"could be",
        r"possibly",
        r"I believe,? (but|though)",
        r"if I('m| am) not mistaken",
        r"I('d| would) guess",
        r"don't quote me",
        r"take this with",
    ]

    # Patterns indicating high confidence
    HIGH_CONFIDENCE_PATTERNS = [
        r"definitely",
        r"certainly",
        r"I('m| am) (very |quite )?confident",
        r"without (a )?doubt",
        r"clearly",
        r"obviously",
        r"absolutely",
    ]

    # Patterns indicating knowledge limits
    KNOWLEDGE_LIMIT_PATTERNS = [
        r"I don't (really )?know",
        r"I('m| am) not familiar",
        r"outside (of )?my (knowledge|expertise)",
        r"I (can't|cannot) (access|verify)",
        r"my (knowledge|training) (cutoff|ends)",
        r"I don't have (current|recent|up-to-date)",
    ]

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "confidence.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Compile patterns
        self._low_conf_re = re.compile(
            '|'.join(self.LOW_CONFIDENCE_PATTERNS),
            re.IGNORECASE
        )
        self._high_conf_re = re.compile(
            '|'.join(self.HIGH_CONFIDENCE_PATTERNS),
            re.IGNORECASE
        )
        self._limit_re = re.compile(
            '|'.join(self.KNOWLEDGE_LIMIT_PATTERNS),
            re.IGNORECASE
        )

    def _init_db(self):
        """Initialize the confidence database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confidence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicted_confidence REAL NOT NULL,
                    uncertainty_type TEXT NOT NULL,
                    reasoning TEXT,
                    factors TEXT,
                    actual_outcome REAL,
                    verified_at TEXT,
                    calibration_error REAL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conf_timestamp
                ON confidence_records(timestamp DESC)
            """)
            conn.commit()

    def assess_confidence(
        self,
        response: str,
        task_type: str = "general",
        context: str = ""
    ) -> ConfidenceScore:
        """
        Assess confidence in a response based on linguistic signals.

        Returns a ConfidenceScore with reasoning.
        """
        factors = []
        score = 0.5  # Default neutral

        # Check for low confidence signals
        low_matches = len(self._low_conf_re.findall(response))
        if low_matches > 0:
            score -= 0.1 * min(low_matches, 3)
            factors.append(f"low_confidence_language ({low_matches} signals)")

        # Check for high confidence signals
        high_matches = len(self._high_conf_re.findall(response))
        if high_matches > 0:
            score += 0.1 * min(high_matches, 3)
            factors.append(f"high_confidence_language ({high_matches} signals)")

        # Check for knowledge limit signals
        limit_matches = len(self._limit_re.findall(response))
        if limit_matches > 0:
            score -= 0.2 * min(limit_matches, 2)
            factors.append(f"knowledge_limit_expressed ({limit_matches} signals)")

        # Adjust based on task type (some tasks are inherently more uncertain)
        task_adjustments = {
            "factual": 0.0,
            "code": 0.0,
            "creative": -0.1,
            "prediction": -0.2,
            "speculation": -0.3,
            "current_events": -0.2,
        }
        if task_type in task_adjustments:
            adj = task_adjustments[task_type]
            if adj != 0:
                score += adj
                factors.append(f"task_type_adjustment ({task_type}: {adj:+.1f})")

        # Response length can indicate certainty (very short might be uncertain)
        word_count = len(response.split())
        if word_count < 10:
            factors.append("short_response")
        elif word_count > 200:
            score += 0.05
            factors.append("detailed_response")

        # Question marks in response (not rhetorical) indicate uncertainty
        question_count = response.count('?')
        if question_count > 1:
            score -= 0.05 * min(question_count - 1, 3)
            factors.append(f"questions_in_response ({question_count})")

        # Clamp to valid range
        score = max(0.05, min(0.95, score))

        # Determine uncertainty type
        if limit_matches > 0:
            uncertainty_type = UncertaintyType.EPISTEMIC
        elif "prediction" in task_type or "future" in context.lower():
            uncertainty_type = UncertaintyType.ALEATORIC
        else:
            uncertainty_type = UncertaintyType.MODEL

        reasoning = self._generate_reasoning(score, factors)

        return ConfidenceScore(
            score=round(score, 2),
            uncertainty_type=uncertainty_type.value,
            reasoning=reasoning,
            factors=factors
        )

    def _generate_reasoning(self, score: float, factors: List[str]) -> str:
        """Generate human-readable reasoning for the confidence score"""
        if score >= 0.8:
            level = "high"
            desc = "The response shows clear confidence and specificity."
        elif score >= 0.6:
            level = "moderate"
            desc = "The response is reasonably confident with some hedging."
        elif score >= 0.4:
            level = "uncertain"
            desc = "The response shows notable uncertainty."
        else:
            level = "low"
            desc = "The response indicates significant uncertainty or knowledge gaps."

        factors_str = ", ".join(factors[:3]) if factors else "no specific signals"
        return f"Confidence: {level} ({score:.0%}). {desc} Factors: {factors_str}."

    def record_prediction(
        self,
        subject: str,
        confidence: ConfidenceScore
    ) -> int:
        """Record a confidence prediction for later verification"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO confidence_records
                (timestamp, subject, predicted_confidence, uncertainty_type,
                 reasoning, factors)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                subject[:500],
                confidence.score,
                confidence.uncertainty_type,
                confidence.reasoning,
                json.dumps(confidence.factors)
            ))
            conn.commit()
            return cursor.lastrowid

    def verify_prediction(
        self,
        record_id: int,
        was_correct: bool
    ) -> float:
        """
        Verify a prediction and record the calibration error.

        Returns the calibration error (positive = overconfident)
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT predicted_confidence FROM confidence_records WHERE id = ?",
                (record_id,)
            ).fetchone()

            if not row:
                return 0.0

            predicted = row[0]
            actual = 1.0 if was_correct else 0.0
            calibration_error = predicted - actual

            conn.execute("""
                UPDATE confidence_records
                SET actual_outcome = ?, verified_at = ?, calibration_error = ?
                WHERE id = ?
            """, (
                actual,
                datetime.now().isoformat(),
                calibration_error,
                record_id
            ))
            conn.commit()

            return calibration_error

    def get_calibration_stats(self, days: int = 30) -> CalibrationStats:
        """Get calibration statistics for recent predictions"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT predicted_confidence, actual_outcome
                FROM confidence_records
                WHERE verified_at IS NOT NULL AND timestamp > ?
            """, (cutoff,)).fetchall()

            if not rows:
                return CalibrationStats(
                    total_predictions=0,
                    avg_confidence=0.0,
                    avg_accuracy=0.0,
                    calibration_error=0.0,
                    overconfidence_rate=0.0,
                    underconfidence_rate=0.0,
                    by_bucket={}
                )

            predictions = [r[0] for r in rows]
            outcomes = [r[1] for r in rows]

            avg_conf = statistics.mean(predictions)
            avg_acc = statistics.mean(outcomes)
            cal_error = avg_conf - avg_acc

            # Count over/under confidence
            overconfident = sum(1 for p, o in rows if p > o + 0.1)
            underconfident = sum(1 for p, o in rows if p < o - 0.1)

            # Bucket by confidence level
            buckets = {
                "0.0-0.2": {"predictions": [], "outcomes": []},
                "0.2-0.4": {"predictions": [], "outcomes": []},
                "0.4-0.6": {"predictions": [], "outcomes": []},
                "0.6-0.8": {"predictions": [], "outcomes": []},
                "0.8-1.0": {"predictions": [], "outcomes": []},
            }

            for pred, out in rows:
                if pred < 0.2:
                    bucket = "0.0-0.2"
                elif pred < 0.4:
                    bucket = "0.2-0.4"
                elif pred < 0.6:
                    bucket = "0.4-0.6"
                elif pred < 0.8:
                    bucket = "0.6-0.8"
                else:
                    bucket = "0.8-1.0"

                buckets[bucket]["predictions"].append(pred)
                buckets[bucket]["outcomes"].append(out)

            by_bucket = {}
            for bucket, data in buckets.items():
                if data["predictions"]:
                    by_bucket[bucket] = {
                        "count": len(data["predictions"]),
                        "avg_confidence": statistics.mean(data["predictions"]),
                        "avg_accuracy": statistics.mean(data["outcomes"]),
                        "calibration_error": statistics.mean(data["predictions"]) - statistics.mean(data["outcomes"])
                    }

            return CalibrationStats(
                total_predictions=len(rows),
                avg_confidence=avg_conf,
                avg_accuracy=avg_acc,
                calibration_error=cal_error,
                overconfidence_rate=overconfident / len(rows),
                underconfidence_rate=underconfident / len(rows),
                by_bucket=by_bucket
            )

    def get_confidence_adjustment(self) -> float:
        """
        Get a suggested adjustment to confidence scores based on calibration.

        Positive = should be more confident
        Negative = should be less confident
        """
        stats = self.get_calibration_stats()

        if stats.total_predictions < 10:
            return 0.0  # Not enough data

        # If we're overconfident, suggest lowering scores
        # If underconfident, suggest raising
        return -stats.calibration_error * 0.5  # Partial correction

    def should_express_uncertainty(self, confidence: ConfidenceScore) -> bool:
        """Determine if RAEC should explicitly express uncertainty"""
        # Express uncertainty if:
        # 1. Confidence is below threshold
        # 2. It's epistemic uncertainty (could learn more)
        # 3. The calibration history suggests overconfidence

        if confidence.score < 0.5:
            return True

        if confidence.uncertainty_type == UncertaintyType.EPISTEMIC.value:
            return confidence.score < 0.7

        adjustment = self.get_confidence_adjustment()
        adjusted = confidence.score + adjustment

        return adjusted < 0.5

    def format_calibration_report(self) -> str:
        """Format calibration statistics for display"""
        stats = self.get_calibration_stats()

        if stats.total_predictions == 0:
            return "No verified predictions yet. Calibration data will build over time."

        lines = [
            "Confidence Calibration Report",
            "=" * 40,
            f"Total verified predictions: {stats.total_predictions}",
            f"Average confidence: {stats.avg_confidence:.1%}",
            f"Actual accuracy: {stats.avg_accuracy:.1%}",
            f"Calibration error: {stats.calibration_error:+.1%}",
            "",
            f"Overconfidence rate: {stats.overconfidence_rate:.1%}",
            f"Underconfidence rate: {stats.underconfidence_rate:.1%}",
        ]

        if stats.by_bucket:
            lines.append("")
            lines.append("By confidence level:")
            for bucket, data in sorted(stats.by_bucket.items()):
                lines.append(
                    f"  {bucket}: {data['count']} predictions, "
                    f"accuracy {data['avg_accuracy']:.1%}"
                )

        # Recommendation
        lines.append("")
        if stats.calibration_error > 0.1:
            lines.append("Recommendation: Be more conservative with confidence estimates.")
        elif stats.calibration_error < -0.1:
            lines.append("Recommendation: You can trust your assessments more.")
        else:
            lines.append("Calibration is good. Confidence estimates are accurate.")

        return "\n".join(lines)
