import numpy as np
import joblib
import torch


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
        low_threshold=0.40,
        high_threshold=0.50
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
        # GPU availability
        # -------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BinaryClassifier] Using device: {self.device}")

        # -------------------------
        # Load models
        # -------------------------
        self.lr_model = joblib.load(lr_model_path)
        self.xgb_model, self.encoder = joblib.load(xgb_model_path)
        
        # Move encoder to GPU if available
        if self.device == "cuda":
            try:
                self.encoder = self.encoder.to(self.device)
            except:
                pass  # Some models don't support .to()

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
        # Use GPU for encoding with larger batch size
        batch_size = 128 if self.device == "cuda" else 64
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            device=self.device
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
    # Human-readable output with confidence score
    # --------------------------------------------------
    def predict_with_labels(self, texts):
        """
        Returns list of dicts with predicted label and confidence score
        
        For each text, returns:
        {
            "label": "SAFE|MAYBE_HATE|HATE",
            "confidence": float (0-1)
        }
        """
        probs = self.predict_proba(texts)
        labels = self.predict(texts)

        output = []
        for p, l in zip(probs, labels):
            if l == 0:
                tag = "SAFE"
                confidence = 1.0 - p
            elif l == 1:
                tag = "MAYBE_HATE"
                confidence = 0.5  # Middle ground
            else:  # l == 2
                tag = "HATE"
                confidence = p

            output.append({
                "label": tag,
                "confidence": round(float(confidence), 4)
            })

        return output
