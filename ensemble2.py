# hatespeech/ensemble2.py
import os
import joblib
import numpy as np
import xgboost as xgb
import re
from sentence_transformers import SentenceTransformer

# small helper lists (extend as needed)
_REPORTING_VERBS = {
    "said", "stated", "reported", "claimed", "wrote", "tweeted", "noted",
    "added", "commented", "explained", "claimed", "noted", "replied"
}
_CONDEMN_VERBS = {"condemn", "condemned", "denounced", "criticized", "criticise"}
_DISCLAIMER_PHRASES = {
    "i'm not racist", "i'm not racist but", "not racist but", "not to be racist",
    "i am not racist but", "i'm not saying", "just asking", "honest question"
}
_DOGWHISTLE_PATTERNS = [
    r"\bthey (don't|dont) belong\b",
    r"\b(our|my) country was better\b",
    r"\b(keep|preserve) (our|the) (culture|country|heritage|purity)\b",
    r"\breplace them\b",
    r"\bgo back to where they came\b"
]


def _contains_pattern_list(text, pattern_list):
    t = text.lower()
    for p in pattern_list:
        if re.search(p, t):
            return True
    return False


def _word_in_text(text, wordset):
    t = text.lower()
    for w in wordset:
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            return True
    return False


class HateSpeechEnsemble:
    """
    Ensemble wrapper with:
      - robust model loading (handles pipelines, tuples, xgboost Booster)
      - rule layer (quotes, reporting verbs, disclaimers, dogwhistles, negation)
      - optional calibrator (joblib saved at models/calibrator.joblib)
      - debug prediction output with per-model probs and flags
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # load lr pipeline (sklearn pipeline saved as single object)
        self.lr = joblib.load(os.path.join(model_dir, "lr_tfidf_flagged.joblib"))

        # load xgb_tfidf - may be pipeline or tuple (model, vectorizer)
        obj = joblib.load(os.path.join(model_dir, "xgb_tfidf_flagged.joblib"))
        if isinstance(obj, tuple) and len(obj) >= 2:
            self.xgb_tfidf_model = obj[0]
            self.xgb_tfidf_vectorizer = obj[1]
        else:
            # pipeline saved as one object (estimator supports predict_proba on raw text)
            self.xgb_tfidf_model = obj
            self.xgb_tfidf_vectorizer = None

        # load xgb_semantic - expected saved as (model, encoder)
        obj = joblib.load(os.path.join(model_dir, "xgb_semantic_flagged.joblib"))
        if isinstance(obj, tuple) and len(obj) >= 2:
            self.xgb_semantic_model = obj[0]
            self.encoder = obj[1]
        else:
            # defensive: if single object, assume it's (model) and encoder separately saved (unlikely)
            raise ValueError("xgb_semantic_flagged.joblib must be a tuple (model, encoder)")

        # move encoder to device if supported
        try:
            self.encoder.to("cuda")
        except Exception:
            # ignore if not supported
            pass

        # optional calibrator
        calibrator_path = os.path.join(model_dir, "calibrator.joblib")
        if os.path.exists(calibrator_path):
            self.calibrator = joblib.load(calibrator_path)
            self.has_calibrator = True
        else:
            self.calibrator = None
            self.has_calibrator = False

        # ensemble configuration (editable)
        self.w_lr = 0.30
        self.w_xgb_tfidf = 0.30
        self.w_xgb_semantic = 0.40

        # thresholds used to derive labels (applied on calibrated prob if calibrator exists)
        self.low_th = 0.40
        self.high_th = 0.55

    # -------------------------
    # Rule layer: flags + multiplier
    # -------------------------
    def _rule_flags_and_multiplier(self, text: str):
        """
        Returns (flags_dict, multiplier_float)
        multiplier >1 increases probability, <1 decreases.
        """
        flags = {}
        t = text.strip()
        lower = t.lower()

        # quotes detection
        in_quotes = bool(re.search(r'["“”\'].*?["“”\']', t))
        flags["in_quotes"] = in_quotes

        # reporting verbs
        has_reporting = _word_in_text(t, _REPORTING_VERBS)
        flags["has_reporting_verb"] = has_reporting

        # condemnation verbs (reporting + condemn suggests reported condemnation)
        has_condemn = _word_in_text(t, _CONDEMN_VERBS)
        flags["has_condemn_verb"] = has_condemn

        # disclaimer
        has_disclaimer = any(phrase in lower for phrase in _DISCLAIMER_PHRASES)
        flags["has_disclaimer"] = has_disclaimer

        # dogwhistle
        has_dogwhistle = _contains_pattern_list(t, _DOGWHISTLE_PATTERNS)
        flags["has_dogwhistle"] = has_dogwhistle

        # negation simple check (not perfect)
        has_negation = bool(re.search(r"\b(no|not|never|n't|never)\b", lower))
        flags["has_negation"] = has_negation

        # short heuristic multipliers (tune on validation)
        multiplier = 1.0

        # if content is quoted AND reporting verb present -> down-weight (reported speech)
        if flags["in_quotes"] and flags["has_reporting_verb"]:
            multiplier *= 0.5  # reduce confidence

        # if condemnation verb present and reporting -> reduce (it's condemnation, not endorsement)
        if flags["has_condemn_verb"] and flags["has_reporting_verb"]:
            multiplier *= 0.6

        # if explicit disclaimer present, slightly increase chance (disclaimer often precedes subtle hate)
        if flags["has_disclaimer"]:
            multiplier *= 1.15

        # dogwhistle -> increase weight (these are often implicit hate)
        if flags["has_dogwhistle"]:
            multiplier *= 1.25

        # negation near hate markers -> reduce (e.g., "not a racist")
        if flags["has_negation"] and not flags["has_dogwhistle"]:
            multiplier *= 0.8

        # clip multiplier to reasonable range
        multiplier = float(max(0.4, min(1.6, multiplier)))

        return flags, multiplier

    # -------------------------
    # per-model prob helpers (defensive)
    # -------------------------
    def _lr_prob(self, texts):
        # pipeline supports raw text
        return self.lr.predict_proba(texts)[:, 1]

    def _xgb_tfidf_prob(self, texts):
        # if we have separate model+vectorizer, transform then predict
        if self.xgb_tfidf_vectorizer is not None:
            X = self.xgb_tfidf_vectorizer.transform(texts)
            # model may be sklearn estimator or xgboost Booster
            if hasattr(self.xgb_tfidf_model, "predict_proba"):
                return self.xgb_tfidf_model.predict_proba(X)[:, 1]
            else:
                # assume Booster or raw xgboost; create DMatrix
                dmat = xgb.DMatrix(X)
                preds = self.xgb_tfidf_model.predict(dmat)
                return np.array(preds).astype(float)
        else:
            # saved pipeline supports raw text
            if hasattr(self.xgb_tfidf_model, "predict_proba"):
                return self.xgb_tfidf_model.predict_proba(texts)[:, 1]
            else:
                # if it's a Booster, we must vectorize; impossible without vectorizer
                raise ValueError("xgb_tfidf was saved as Booster but no vectorizer found.")

    def _xgb_semantic_prob(self, texts):
        # encoder returns numpy embeddings
        emb = self.encoder.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        # model may be sklearn XGBClassifier (has predict_proba) or Booster
        if hasattr(self.xgb_semantic_model, "predict_proba"):
            try:
                return self.xgb_semantic_model.predict_proba(emb)[:, 1]
            except Exception:
                # fallback to dmatrix if predict_proba fails
                dmat = xgb.DMatrix(emb)
                return np.array(self.xgb_semantic_model.predict(dmat)).astype(float)
        else:
            dmat = xgb.DMatrix(emb)
            return np.array(self.xgb_semantic_model.predict(dmat)).astype(float)

    # -------------------------
    # calibration helper (platt/logistic calibrator)
    # -------------------------
    def _apply_calibrator(self, probs):
        if not self.has_calibrator or self.calibrator is None:
            return probs
        # calibrator expects shape (n_samples, 1)
        X = np.array(probs).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(X)[:, 1]
        return calibrated

    # -------------------------
    # Public: predict (simple) and predict_debug
    # -------------------------
    def predict(self, texts):
        """
        Returns list of dicts: { "label":..., "confidence":... }
        label/thresholding uses calibrated prob if calibrator present else raw final prob
        """
        debug_out = self.predict_debug(texts)
        results = []
        for d in debug_out:
            prob_to_use = d["calibrated_prob"] if d.get("calibrated_prob") is not None else d["final_prob"]
            if prob_to_use >= self.high_th:
                label = "HATE"
            elif prob_to_use >= self.low_th:
                label = "MAYBE_HATE"
            else:
                label = "SAFE"
            results.append({"label": label, "confidence": float(prob_to_use)})
        return results

    def predict_debug(self, texts):
        """
        Returns detailed per-sample info:
        {
           "text": ..., 
           "p_lr": ..., "p_xgb_tfidf": ..., "p_xgb_semantic": ...,
           "flags": {...}, "multiplier": float,
           "raw_final_prob": float,
           "calibrated_prob": float or None
        }
        """
        texts = [str(t) for t in texts]
        # compute per-model probs
        p_lr = self._lr_prob(texts)
        p_xgb_tfidf = self._xgb_tfidf_prob(texts)
        p_xgb_sem = self._xgb_semantic_prob(texts)

        out = []
        for i, txt in enumerate(texts):
            # per-sample rule flags
            flags, multiplier = self._rule_flags_and_multiplier(txt)

            # weighted ensemble raw prob
            raw_final = (
                self.w_lr * float(p_lr[i])
                + self.w_xgb_tfidf * float(p_xgb_tfidf[i])
                + self.w_xgb_semantic * float(p_xgb_sem[i])
            )

            # apply multiplier from rule layer
            final_prob = float(max(0.0, min(1.0, raw_final * multiplier)))

            # apply calibrator
            if self.has_calibrator:
                calibrated = float(self._apply_calibrator([final_prob])[0])
            else:
                calibrated = None

            out.append({
                "text": txt,
                "p_lr": float(p_lr[i]),
                "p_xgb_tfidf": float(p_xgb_tfidf[i]),
                "p_xgb_semantic": float(p_xgb_sem[i]),
                "flags": flags,
                "multiplier": multiplier,
                "raw_ensemble_prob": float(raw_final),
                "final_prob": final_prob,
                "calibrated_prob": calibrated,
            })
        return out
