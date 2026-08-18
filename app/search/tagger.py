"""Role-family and seniority tagging by weak supervision.

Nobody is going to hand-label ten thousand job postings for a take-home. The
alternative that actually works is Snorkel-style weak supervision: write a
handful of high-precision labelling functions, accept that they only fire on
part of the corpus, and train a classifier on what they produce so it
generalises to the rest.

The honest framing, which is also what the dashboard shows: the reported score
is *agreement with the labelling functions on held-out rows*, not accuracy
against human ground truth. It measures whether the model learned the rules'
generalisation, not whether the rules are right. Calling that "94% accurate"
would be the kind of number this brief specifically asks candidates not to
invent.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from .. import db

# --- labelling functions --------------------------------------------------
# Each is (label, regex). Ordered: the first match wins, so the more specific
# patterns are listed first.

ROLE_RULES: tuple[tuple[str, str], ...] = (
    ("ml-ai",       r"\b(machine learning|ml engineer|mlops|deep learning|nlp|"
                    r"computer vision|llm|genai|generative ai|ai engineer|"
                    r"research engineer|applied scientist)\b"),
    ("data",        r"\b(data scientist|data engineer|analytics engineer|"
                    r"data analyst|bi (developer|analyst)|etl|dbt|warehouse)\b"),
    ("infra",       r"\b(sre|site reliability|devops|platform engineer|"
                    r"infrastructure|kubernetes|cloud engineer)\b"),
    ("frontend",    r"\b(frontend|front-end|react|vue|angular|ui engineer|"
                    r"web developer)\b"),
    ("mobile",      r"\b(ios|android|react native|flutter|mobile engineer)\b"),
    ("security",    r"\b(security engineer|appsec|infosec|penetration|soc analyst)\b"),
    ("backend",     r"\b(backend|back-end|api engineer|golang|java developer|"
                    r"python (engineer|developer)|node\.?js|rails|django)\b"),
    ("product",     r"\b(product manager|program manager|product owner)\b"),
    ("design",      r"\b(designer|ux|ui/ux|product design)\b"),
    ("sales-ops",   r"\b(sales|account executive|recruiter|marketing|"
                    r"customer success|support)\b"),
)

SENIORITY_RULES: tuple[tuple[str, str], ...] = (
    ("lead",     r"\b(principal|staff|distinguished|head of|director|vp|"
                 r"lead engineer|tech lead|architect)\b"),
    ("senior",   r"\b(senior|sr\.?|iii|experienced)\b"),
    ("junior",   r"\b(junior|jr\.?|graduate|entry[- ]level|associate|intern|"
                 r"trainee|fresher)\b"),
    ("mid",      r"\b(engineer ii|mid[- ]level|software engineer 2)\b"),
)

_ROLE_COMPILED = [(lbl, re.compile(pat, re.I)) for lbl, pat in ROLE_RULES]
_SENIORITY_COMPILED = [(lbl, re.compile(pat, re.I)) for lbl, pat in SENIORITY_RULES]

_KV = "tagger_metrics"
_MIN_PER_CLASS = 3      # below this a class cannot be cross-validated
_MIN_CONFIDENCE = 0.60  # below this the model declines to label


def _label_text(row: dict) -> str:
    """What the labelling functions see: title and tags only.

    Descriptions are actively harmful here. A backend role whose blurb mentions
    "data warehouse" matches the `data` rule and gets mislabelled -- observed on
    real ingested rows before this was narrowed. The role lives in the title;
    the description is context, not evidence.
    """
    tags = " ".join(json.loads(row.get("tags") or "[]"))
    return " ".join(filter(None, [row.get("title") or "", tags]))


def _feature_text(row: dict) -> str:
    """What the *classifier* sees: a little more context than the rules get.

    The model can use the description because it weighs evidence rather than
    first-match-wins, and because the rules -- which do not see it -- always
    override it where they fired.
    """
    tags = " ".join(json.loads(row.get("tags") or "[]"))
    return " ".join(filter(None, [
        row.get("title") or "", row.get("title") or "",
        tags, (row.get("description") or "")[:300],
    ]))


def label_role(text: str) -> str | None:
    for label, pattern in _ROLE_COMPILED:
        if pattern.search(text):
            return label
    return None


def label_seniority(text: str) -> str | None:
    for label, pattern in _SENIORITY_COMPILED:
        if pattern.search(text):
            return label
    # A title with none of the markers is almost always mid-level, but that is
    # an inference, not a rule match -- so it stays unlabelled and the model is
    # left to decide.
    return None


class WeakTagger:
    """Rules label what they can; a classifier covers the rest."""

    def __init__(self) -> None:
        self.role_model = None
        self.seniority_model = None
        self.metrics: dict[str, object] = {}

    @staticmethod
    def _pipeline():
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                            strip_accents="unicode", max_features=40_000),
            LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0),
        )

    def fit_and_apply(self) -> dict[str, object]:
        rows = db.query(
            "SELECT canonical_id, title, company, description, tags FROM jobs LIMIT 8000"
        )
        if len(rows) < 40:
            return {"trained": False, "reason": f"only {len(rows)} rows"}

        label_texts = [_label_text(r) for r in rows]
        texts = [_feature_text(r) for r in rows]
        roles = [label_role(t) for t in label_texts]
        seniorities = [label_seniority(t) for t in label_texts]

        out: dict[str, object] = {
            "trained": True,
            "n_rows": len(rows),
            "role_coverage": round(sum(r is not None for r in roles) / len(rows), 3),
            "seniority_coverage": round(
                sum(s is not None for s in seniorities) / len(rows), 3
            ),
            "note": (
                "Scores are agreement with the labelling functions on held-out "
                "rows, not accuracy against human labels."
            ),
            "trained_at": time.time(),
        }

        predictions: dict[str, tuple[list[str], list[str]]] = {}
        for field, labels, attr in (
            ("role_family", roles, "role_model"),
            ("seniority", seniorities, "seniority_model"),
        ):
            idx = [i for i, lbl in enumerate(labels) if lbl is not None]
            counts = {c: [labels[i] for i in idx].count(c)
                      for c in {labels[i] for i in idx}}
            # Drop classes too rare to cross-validate rather than abandoning the
            # whole field. One board posting a single `mobile` role must not
            # cost every other row its tag -- and those rare rows keep their
            # rule label regardless, since rules always win where they fired.
            trainable = {c for c, n in counts.items() if n >= _MIN_PER_CLASS}
            out[f"{field}_classes_kept"] = sorted(trainable)
            out[f"{field}_classes_too_rare"] = sorted(set(counts) - trainable)
            if len(trainable) < 2:
                out[f"{field}_cv"] = None
                continue

            keep = [i for i in idx if labels[i] in trainable]
            X = [texts[i] for i in keep]
            y = [labels[i] for i in keep]
            folds = min(3, min(counts[c] for c in trainable))
            try:
                scores = cross_val_score(self._pipeline(), X, y, cv=folds)
                out[f"{field}_cv"] = round(float(np.mean(scores)), 3)
            except ValueError:
                out[f"{field}_cv"] = None
            model = self._pipeline()
            model.fit(X, y)
            setattr(self, attr, model)
            # Only take a model prediction it is actually confident about.
            # Without this the classifier labels every unmarked title "senior"
            # or "lead" simply because those are the classes it has seen most,
            # which turns an honest null into a confident guess. An unlabelled
            # row is a better answer than a wrong one.
            proba = model.predict_proba(texts)
            classes = list(model.classes_)
            guesses = [
                classes[int(row.argmax())] if row.max() >= _MIN_CONFIDENCE else None
                for row in proba
            ]
            out[f"{field}_model_filled"] = sum(g is not None for g in guesses)
            predictions[field] = (guesses, y)

        # Rule wins where it fired; model fills the gap. Precision over recall:
        # a hand-written rule that matched is better evidence than a model that
        # extrapolated.
        role_pred = predictions.get("role_family", ([None] * len(rows), []))[0]
        sen_pred = predictions.get("seniority", ([None] * len(rows), []))[0]
        updates = [
            (roles[i] or role_pred[i], seniorities[i] or sen_pred[i],
             rows[i]["canonical_id"])
            for i in range(len(rows))
        ]
        db.executemany(
            "UPDATE jobs SET role_family = ?, seniority = ? WHERE canonical_id = ?",
            updates,
        )

        self.metrics = out
        db.kv_set(_KV, out)
        return out

    def status(self) -> dict[str, object]:
        return self.metrics or db.kv_get(_KV, {"trained": False})


tagger = WeakTagger()
