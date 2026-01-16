import numpy as np
import joblib


class BinaryClassifier:
    """
    Final hate-speech classifier using soft-decision ensemble
    with tri-level output:

        SAFE        : p <= low
        MAYBE_HATE  : low < p <= high
        HATE        : p > high
    """

    def __init__(
        self,
        lr_model_path,
        xgb_model_path,
        semantic_weight=0.7,
        lexical_weight=0.3,
        low_threshold=0.5,
        high_threshold=0.7
    ):
        # -------------------------
        # Validate ensemble weights
        # -------------------------
        if not np.isclose(semantic_weight + lexical_weight, 1.0):
            raise ValueError("Ensemble weights must sum to 1")

        # -------------------------
        # Validate thresholds
        # -------------------------
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 <= low < high <= 1"
            )

        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.low = low_threshold
        self.high = high_threshold

        # -------------------------
        # Load models
        # -------------------------
        self.lr_model = joblib.load(lr_model_path)
        self.xgb_model, self.encoder = joblib.load(xgb_model_path)

    # --------------------------------------------------
    # Internal probability sources
    # --------------------------------------------------
    def _lr_probability(self, texts):
        """
        P(hate | lexical features)
        """
        return self.lr_model.predict_proba(texts)[:, 1]

    def _xgb_probability(self, texts):
        """
        P(hate | semantic embeddings)
        """
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False
        )
        return self.xgb_model.predict_proba(embeddings)[:, 1]

    # --------------------------------------------------
    # Soft-decision probability
    # --------------------------------------------------
    def predict_proba(self, texts):
        """
        Returns final soft-decision probability:
            p = w_lexical * p_lr + w_semantic * p_xgb
        """
        lr_probs = self._lr_probability(texts)
        xgb_probs = self._xgb_probability(texts)

        if len(lr_probs) != len(xgb_probs):
            raise RuntimeError("Probability size mismatch")

        final_probs = (
            self.lexical_weight * lr_probs +
            self.semantic_weight * xgb_probs
        )

        return final_probs

    # --------------------------------------------------
    # Tri-level decision logic
    # --------------------------------------------------
    def predict(self, texts):
        """
        Returns numeric labels:
            0 -> SAFE
            1 -> MAYBE_HATE
            2 -> HATE
        """
        probs = self.predict_proba(texts)
        labels = np.zeros(len(probs), dtype=int)

        labels[(probs > self.low) & (probs <= self.high)] = 1
        labels[probs > self.high] = 2

        return labels

    # --------------------------------------------------
    # Human-readable output
    # --------------------------------------------------
    def predict_with_labels(self, texts):
        """
        Returns list of dicts with label + probability
        """
        probs = self.predict_proba(texts)
        labels = self.predict(texts)

        output = []
        for p, l in zip(probs, labels):
            if l == 0:
                tag = "SAFE"
            elif l == 1:
                tag = "MAYBE_HATE"
            else:
                tag = "HATE"

            output.append({
                "label": tag,
                "probability": float(p)
            })

        return output
