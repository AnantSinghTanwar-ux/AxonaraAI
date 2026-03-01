"""
Axonara - Cognitive Load Detection Model
=========================================
Simulates cognitive overload detection using typing_speed, pause_duration,
and edit_frequency. A logistic regression classifier estimates whether a
learner is experiencing cognitive overload.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Synthetic training data generator
# ---------------------------------------------------------------------------

def _generate_training_data(n_samples: int = 500, seed: int = 42):
    """
    Create synthetic data that mimics real learner behaviour signals.

    Features (per sample):
        typing_speed   – words per minute  (lower → higher load)
        pause_duration – seconds of inactivity (higher → higher load)
        edit_frequency – edits per minute   (higher → higher load)

    Label:
        0 = normal cognitive load
        1 = cognitive overload
    """
    rng = np.random.RandomState(seed)

    # --- Normal load samples (label 0) ---
    n_normal = n_samples // 2
    normal = np.column_stack([
        rng.normal(60, 12, n_normal),   # typing_speed  ~60 wpm
        rng.normal(2, 0.8, n_normal),   # pause_duration ~2 s
        rng.normal(3, 1.0, n_normal),   # edit_frequency ~3/min
    ])

    # --- Overload samples (label 1) ---
    n_overload = n_samples - n_normal
    overload = np.column_stack([
        rng.normal(25, 10, n_overload),  # typing drops to ~25 wpm
        rng.normal(8, 2.5, n_overload),  # pauses rise to ~8 s
        rng.normal(10, 3.0, n_overload), # edits spike to ~10/min
    ])

    X = np.vstack([normal, overload])
    y = np.array([0] * n_normal + [1] * n_overload)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# CognitiveLoadModel
# ---------------------------------------------------------------------------

class CognitiveLoadModel:
    """Logistic-regression estimator for cognitive overload."""

    OVERLOAD_THRESHOLD = 0.55  # probability above which we flag overload

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self._is_trained = False
        self._train()

    # ---- internal -----------------------------------------------------------

    def _train(self):
        X, y = _generate_training_data()
        self.model.fit(X, y)
        self._is_trained = True

    # ---- public API ---------------------------------------------------------

    def predict(
        self,
        typing_speed: float,
        pause_duration: float,
        edit_frequency: float,
    ) -> dict:
        """
        Predict cognitive load state.

        Returns
        -------
        dict with keys:
            overload        – bool
            overload_score  – float [0, 1]
            status          – str   ("overloaded" | "normal")
            recommendation  – str   (action hint for the frontend)
        """
        features = np.array([[typing_speed, pause_duration, edit_frequency]])
        probability = self.model.predict_proba(features)[0][1]  # P(overload)
        is_overloaded = probability > self.OVERLOAD_THRESHOLD

        return {
            "overload": bool(is_overloaded),
            "overload_score": round(float(probability), 4),
            "status": "overloaded" if is_overloaded else "normal",
            "recommendation": (
                "Learner appears overloaded – switching to simplified content."
                if is_overloaded
                else "Learner is coping well – presenting standard summary."
            ),
        }

    def simulate_random(self, seed: int | None = None) -> dict:
        """Generate random behavioural signals and predict load."""
        rng = np.random.RandomState(seed)
        typing_speed = rng.uniform(15, 80)
        pause_duration = rng.uniform(0.5, 12)
        edit_frequency = rng.uniform(1, 15)
        result = self.predict(typing_speed, pause_duration, edit_frequency)
        result["inputs"] = {
            "typing_speed": round(typing_speed, 2),
            "pause_duration": round(pause_duration, 2),
            "edit_frequency": round(edit_frequency, 2),
        }
        return result


# ---------------------------------------------------------------------------
# Module-level singleton so other modules can import directly
# ---------------------------------------------------------------------------
cognitive_model = CognitiveLoadModel()


if __name__ == "__main__":
    print("=== Cognitive Load Model – quick test ===")
    print(cognitive_model.predict(typing_speed=55, pause_duration=2, edit_frequency=3))
    print(cognitive_model.predict(typing_speed=20, pause_duration=10, edit_frequency=12))
    print(cognitive_model.simulate_random(seed=7))
