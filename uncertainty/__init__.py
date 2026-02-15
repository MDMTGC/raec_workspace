"""
RAEC Uncertainty Quantification - Knowing what you don't know

Tracks and calibrates confidence:
- Explicit confidence scores on outputs
- Calibration tracking (predicted vs actual accuracy)
- Epistemic vs aleatoric uncertainty
- "I don't know" detection
"""

from .confidence import ConfidenceTracker, ConfidenceScore, CalibrationStats

__all__ = [
    'ConfidenceTracker',
    'ConfidenceScore',
    'CalibrationStats',
]
