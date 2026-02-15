"""
Self Evaluator - RAEC tracks its own performance

This is how RAEC learns what it's good and bad at:
- Records outcomes of tasks
- Tracks prediction vs reality (confidence calibration)
- Identifies patterns in failures
- Measures improvement over time
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import statistics


class EvaluationType(Enum):
    """Types of evaluations"""
    TASK_OUTCOME = "task_outcome"           # Did the task succeed?
    PREDICTION = "prediction"                # Was the prediction accurate?
    RESPONSE_QUALITY = "response_quality"    # Was the response good?
    TOOL_USAGE = "tool_usage"                # Did tool use work?
    KNOWLEDGE = "knowledge"                  # Was information accurate?


class OutcomeRating(Enum):
    """How well did something go"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FAILED = 1


@dataclass
class Evaluation:
    """A single evaluation record"""
    id: Optional[int]
    evaluation_type: str
    timestamp: str

    # What was evaluated
    subject: str  # What was being evaluated (task, response, prediction)
    context: str  # Surrounding context

    # Scores
    predicted_confidence: float  # What RAEC thought (0.0-1.0)
    actual_outcome: int          # What actually happened (1-5)
    calibration_error: float     # Difference between prediction and reality

    # Analysis
    success: bool
    failure_reason: Optional[str]
    improvement_note: Optional[str]
    tags: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'Evaluation':
        return cls(
            id=row[0],
            evaluation_type=row[1],
            timestamp=row[2],
            subject=row[3],
            context=row[4],
            predicted_confidence=row[5],
            actual_outcome=row[6],
            calibration_error=row[7],
            success=bool(row[8]),
            failure_reason=row[9],
            improvement_note=row[10],
            tags=json.loads(row[11]) if row[11] else []
        )


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    total_evaluations: int
    success_rate: float
    avg_calibration_error: float
    by_type: Dict[str, Dict[str, float]]
    trend: str  # "improving", "stable", "declining"
    weak_areas: List[str]
    strong_areas: List[str]


class SelfEvaluator:
    """
    RAEC's self-evaluation system.

    Tracks performance, calibrates confidence, identifies weaknesses.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "evaluations.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the evaluations database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    context TEXT,
                    predicted_confidence REAL NOT NULL,
                    actual_outcome INTEGER NOT NULL,
                    calibration_error REAL NOT NULL,
                    success INTEGER NOT NULL,
                    failure_reason TEXT,
                    improvement_note TEXT,
                    tags TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_type
                ON evaluations(evaluation_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_timestamp
                ON evaluations(timestamp DESC)
            """)
            conn.commit()

    def record_evaluation(
        self,
        evaluation_type: EvaluationType,
        subject: str,
        predicted_confidence: float,
        actual_outcome: OutcomeRating,
        context: str = "",
        failure_reason: Optional[str] = None,
        improvement_note: Optional[str] = None,
        tags: List[str] = None
    ) -> Evaluation:
        """Record an evaluation"""
        now = datetime.now().isoformat()

        # Calculate calibration error
        # Map outcome (1-5) to (0.0-1.0) scale for comparison
        outcome_normalized = (actual_outcome.value - 1) / 4.0
        calibration_error = abs(predicted_confidence - outcome_normalized)

        success = actual_outcome.value >= OutcomeRating.ACCEPTABLE.value

        evaluation = Evaluation(
            id=None,
            evaluation_type=evaluation_type.value,
            timestamp=now,
            subject=subject,
            context=context,
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome.value,
            calibration_error=calibration_error,
            success=success,
            failure_reason=failure_reason if not success else None,
            improvement_note=improvement_note,
            tags=tags or []
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO evaluations
                (evaluation_type, timestamp, subject, context, predicted_confidence,
                 actual_outcome, calibration_error, success, failure_reason,
                 improvement_note, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation.evaluation_type,
                evaluation.timestamp,
                evaluation.subject,
                evaluation.context,
                evaluation.predicted_confidence,
                evaluation.actual_outcome,
                evaluation.calibration_error,
                1 if evaluation.success else 0,
                evaluation.failure_reason,
                evaluation.improvement_note,
                json.dumps(evaluation.tags)
            ))
            evaluation.id = cursor.lastrowid
            conn.commit()

        return evaluation

    def record_task_outcome(
        self,
        task: str,
        confidence: float,
        succeeded: bool,
        failure_reason: Optional[str] = None,
        tags: List[str] = None
    ) -> Evaluation:
        """Convenience method for recording task outcomes"""
        outcome = OutcomeRating.GOOD if succeeded else OutcomeRating.FAILED

        return self.record_evaluation(
            evaluation_type=EvaluationType.TASK_OUTCOME,
            subject=task[:200],
            predicted_confidence=confidence,
            actual_outcome=outcome,
            failure_reason=failure_reason,
            tags=tags
        )

    def record_prediction(
        self,
        prediction: str,
        confidence: float,
        was_correct: bool,
        context: str = ""
    ) -> Evaluation:
        """Record a prediction and its accuracy"""
        outcome = OutcomeRating.EXCELLENT if was_correct else OutcomeRating.FAILED

        return self.record_evaluation(
            evaluation_type=EvaluationType.PREDICTION,
            subject=prediction[:200],
            predicted_confidence=confidence,
            actual_outcome=outcome,
            context=context
        )

    def get_recent_evaluations(self, limit: int = 50) -> List[Evaluation]:
        """Get recent evaluations"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM evaluations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [Evaluation.from_row(row) for row in rows]

    def get_evaluations_by_type(
        self,
        eval_type: EvaluationType,
        limit: int = 100
    ) -> List[Evaluation]:
        """Get evaluations of a specific type"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM evaluations
                WHERE evaluation_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (eval_type.value, limit)).fetchall()

            return [Evaluation.from_row(row) for row in rows]

    def get_failure_patterns(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get common failure reasons"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT failure_reason, COUNT(*) as count
                FROM evaluations
                WHERE success = 0 AND failure_reason IS NOT NULL
                GROUP BY failure_reason
                ORDER BY count DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [(row[0], row[1]) for row in rows]

    def get_calibration_stats(self) -> Dict[str, float]:
        """Get confidence calibration statistics"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT predicted_confidence, actual_outcome
                FROM evaluations
                ORDER BY timestamp DESC
                LIMIT 200
            """).fetchall()

            if not rows:
                return {
                    "avg_calibration_error": 0.0,
                    "overconfidence_rate": 0.0,
                    "underconfidence_rate": 0.0
                }

            errors = []
            overconfident = 0
            underconfident = 0

            for pred, actual in rows:
                actual_norm = (actual - 1) / 4.0
                error = pred - actual_norm
                errors.append(abs(error))

                if error > 0.1:
                    overconfident += 1
                elif error < -0.1:
                    underconfident += 1

            return {
                "avg_calibration_error": statistics.mean(errors),
                "overconfidence_rate": overconfident / len(rows),
                "underconfidence_rate": underconfident / len(rows)
            }

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total and success rate
            total = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
            if total == 0:
                return PerformanceMetrics(
                    total_evaluations=0,
                    success_rate=0.0,
                    avg_calibration_error=0.0,
                    by_type={},
                    trend="stable",
                    weak_areas=[],
                    strong_areas=[]
                )

            successes = conn.execute(
                "SELECT COUNT(*) FROM evaluations WHERE success = 1"
            ).fetchone()[0]
            success_rate = successes / total

            # Calibration error
            avg_cal = conn.execute(
                "SELECT AVG(calibration_error) FROM evaluations"
            ).fetchone()[0] or 0.0

            # By type
            by_type = {}
            for et in EvaluationType:
                type_total = conn.execute(
                    "SELECT COUNT(*) FROM evaluations WHERE evaluation_type = ?",
                    (et.value,)
                ).fetchone()[0]

                if type_total > 0:
                    type_success = conn.execute(
                        "SELECT COUNT(*) FROM evaluations WHERE evaluation_type = ? AND success = 1",
                        (et.value,)
                    ).fetchone()[0]

                    by_type[et.value] = {
                        "total": type_total,
                        "success_rate": type_success / type_total,
                        "avg_calibration": conn.execute(
                            "SELECT AVG(calibration_error) FROM evaluations WHERE evaluation_type = ?",
                            (et.value,)
                        ).fetchone()[0] or 0.0
                    }

            # Trend (compare recent to older)
            recent = conn.execute("""
                SELECT AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END)
                FROM evaluations
                ORDER BY timestamp DESC
                LIMIT 50
            """).fetchone()[0] or 0.0

            older = conn.execute("""
                SELECT AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END)
                FROM evaluations
                ORDER BY timestamp DESC
                LIMIT 50 OFFSET 50
            """).fetchone()[0]

            if older is None:
                trend = "stable"
            elif recent > older + 0.1:
                trend = "improving"
            elif recent < older - 0.1:
                trend = "declining"
            else:
                trend = "stable"

            # Weak and strong areas (by tag)
            weak_areas = []
            strong_areas = []

            tags_query = conn.execute("""
                SELECT tags, success FROM evaluations WHERE tags IS NOT NULL
            """).fetchall()

            tag_stats = {}
            for tags_json, success in tags_query:
                try:
                    tags = json.loads(tags_json)
                    for tag in tags:
                        if tag not in tag_stats:
                            tag_stats[tag] = {"success": 0, "total": 0}
                        tag_stats[tag]["total"] += 1
                        if success:
                            tag_stats[tag]["success"] += 1
                except:
                    pass

            for tag, stats in tag_stats.items():
                if stats["total"] >= 3:  # Minimum samples
                    rate = stats["success"] / stats["total"]
                    if rate < 0.5:
                        weak_areas.append(tag)
                    elif rate > 0.8:
                        strong_areas.append(tag)

            return PerformanceMetrics(
                total_evaluations=total,
                success_rate=success_rate,
                avg_calibration_error=avg_cal,
                by_type=by_type,
                trend=trend,
                weak_areas=weak_areas,
                strong_areas=strong_areas
            )

    def get_improvement_suggestions(self) -> List[str]:
        """Generate suggestions based on evaluation patterns"""
        suggestions = []

        # Check calibration
        cal_stats = self.get_calibration_stats()
        if cal_stats["overconfidence_rate"] > 0.3:
            suggestions.append(
                "Consider being more conservative with confidence estimates - "
                "you're often overconfident."
            )
        if cal_stats["underconfidence_rate"] > 0.3:
            suggestions.append(
                "You tend to underestimate your abilities - "
                "trust yourself more on familiar tasks."
            )

        # Check failure patterns
        failures = self.get_failure_patterns(5)
        for reason, count in failures:
            if count >= 3:
                suggestions.append(f"Recurring issue: '{reason}' - investigate and address.")

        # Check weak areas
        metrics = self.get_performance_metrics()
        for area in metrics.weak_areas[:3]:
            suggestions.append(f"Weak area '{area}' needs improvement.")

        if not suggestions:
            suggestions.append("Performance is good. Keep up the solid work.")

        return suggestions

    def get_stats(self) -> dict:
        """Get summary statistics"""
        metrics = self.get_performance_metrics()
        cal_stats = self.get_calibration_stats()

        return {
            "total_evaluations": metrics.total_evaluations,
            "success_rate": metrics.success_rate,
            "avg_calibration_error": metrics.avg_calibration_error,
            "trend": metrics.trend,
            "overconfidence_rate": cal_stats["overconfidence_rate"],
            "weak_areas": metrics.weak_areas,
            "strong_areas": metrics.strong_areas
        }

    def format_performance_report(self) -> str:
        """Format a performance report for display"""
        metrics = self.get_performance_metrics()
        suggestions = self.get_improvement_suggestions()

        lines = [
            "RAEC Performance Report",
            "=" * 40,
            f"Total evaluations: {metrics.total_evaluations}",
            f"Success rate: {metrics.success_rate:.1%}",
            f"Calibration error: {metrics.avg_calibration_error:.2f}",
            f"Trend: {metrics.trend}",
            ""
        ]

        if metrics.weak_areas:
            lines.append(f"Weak areas: {', '.join(metrics.weak_areas)}")
        if metrics.strong_areas:
            lines.append(f"Strong areas: {', '.join(metrics.strong_areas)}")

        lines.append("")
        lines.append("Suggestions:")
        for s in suggestions:
            lines.append(f"  • {s}")

        return "\n".join(lines)
