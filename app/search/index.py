"""Semantic index: search, near-duplicate detection, and "more like this".

Deliberately *not* a transformer. A MiniLM checkpoint plus torch is ~900MB of
image for a corpus of a few thousand short job postings, on a host with 512MB
of RAM. TF-IDF into a truncated SVD gives dense vectors that behave the same
way for this corpus, trains in under a second on every restart, needs no model
download, and can be explained end to end.

Two vectorisers are unioned on purpose:

  word 1-2 grams   the semantics -- "machine learning engineer" vs "ml engineer"
  char 3-5 grams   the robustness -- typos, "K8s"/"Kubernetes", "Sr."/"Senior",
                   and the company-name variants that exact matching misses

SVD on top turns the sparse union into a ~96-dim dense space where cosine
similarity is meaningful and near-duplicate clustering is a threshold rather
than a heuristic.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from ..config import settings
from .. import db
from ..ingest.normalize import norm_company
from .tagger import label_seniority

# Cosine above this, *within one employer*, means "same posting, different
# words".
#
# Calibrated by inspection against 388 live postings, not guessed. At 0.70 it
# starts pairing "Software Engineer" with "Data Engineer" at the same employer,
# which is wrong. At 0.75 the survivors are three pairs that a human agrees
# with. Above 0.80 it misses genuine re-listings that differ by a qualifier
# ("Data Platform Engineer" vs "Data Platform Engineer - Core", 0.78).
#
# Erring low would be the worse mistake: a flagged duplicate hides a real
# vacancy from the reader, while a missed one only leaves the count slightly
# high.
DUP_THRESHOLD = 0.75
_MIN_DOCS = 12          # below this, the SVD has nothing to learn


def _document(row: dict) -> str:
    """What the *search* vectorisers see. Title is repeated because it carries
    far more signal per token than the description, and TF-IDF has no other way
    to know that."""
    tags = " ".join(json.loads(row.get("tags") or "[]"))
    return " ".join(filter(None, [
        row.get("title") or "", row.get("title") or "",
        row.get("company") or "", row.get("location") or "",
        tags, (row.get("description") or "")[:600],
    ]))


def _identity_document(row: dict) -> str:
    """What *duplicate detection* sees: title and location, nothing else.

    Search and dedupe want opposite things from a document, which is why they
    get different ones. Search wants context -- the description is what makes
    "deep learning" find a PyTorch role. Dedupe wants only the fields that
    distinguish one posting from another *at the same employer*, and within one
    employer the description is near-constant boilerplate that pushes every
    cosine toward 1.0. Comparing full documents inside a company block scored
    "Head of Marketing" against "Head of Design" at 0.99. They are not the
    same job.
    """
    return " ".join(filter(None, [
        row.get("title") or "", row.get("location") or "",
    ]))


class SemanticIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self.word_vec: TfidfVectorizer | None = None
        self.char_vec: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.identity_matrix: np.ndarray | None = None
        self.built_at: float = 0.0
        self.n_docs: int = 0
        self.explained_variance: float = 0.0

    @property
    def ready(self) -> bool:
        return self.matrix is not None and len(self.ids) > 0

    def build(self) -> dict[str, object]:
        """Rebuild from scratch. Cheap enough (sub-second for tens of thousands
        of short docs) that incremental updates would be added complexity for
        no measurable win at this scale."""
        rows = db.query(
            "SELECT canonical_id, title, company, location, description, tags "
            "FROM jobs ORDER BY last_seen DESC LIMIT ?",
            (settings.index_corpus_limit,),
        )
        if len(rows) < _MIN_DOCS:
            with self._lock:
                self.ids, self.matrix, self.n_docs = [], None, len(rows)
            self.identity_matrix = None
            return {"built": False, "reason": f"only {len(rows)} documents"}

        docs = [_document(r) for r in rows]
        word_vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True,
            strip_accents="unicode", lowercase=True,
            max_features=settings.tfidf_max_features,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
            lowercase=True, max_features=settings.tfidf_max_features,
        )
        sparse = hstack([word_vec.fit_transform(docs), char_vec.fit_transform(docs)]).tocsr()

        dims = int(min(settings.embedding_dims, max(2, min(sparse.shape) - 1)))
        svd = TruncatedSVD(n_components=dims, random_state=0)
        dense = l2_normalize(svd.fit_transform(sparse))

        # Second, much narrower space, used only for duplicate detection. Kept
        # sparse -- no SVD -- because titles are short and the dimensionality
        # reduction that helps recall in search actively hurts precision here by
        # smearing distinct short titles together.
        #
        # Char n-grams carry most of the weight. On word tokens alone, "Data
        # Platform Engineer" and "Data Platform Engineer - Core" score 0.70,
        # because IDF makes the one rare token ("core") dominate the vector --
        # which is backwards for this job, where a shared prefix is exactly the
        # evidence we want. Character overlap is not fooled that way.
        ident_docs = [_identity_document(r) for r in rows]
        ident_word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            strip_accents="unicode", lowercase=True,
        )
        ident_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
            lowercase=True,
        )
        identity = l2_normalize(hstack([
            ident_word.fit_transform(ident_docs),
            ident_char.fit_transform(ident_docs),
        ]).tocsr())

        with self._lock:
            self.ids = [r["canonical_id"] for r in rows]
            self.matrix = dense
            self.identity_matrix = identity
            self.word_vec, self.char_vec, self.svd = word_vec, char_vec, svd
            self.built_at = time.time()
            self.n_docs = len(rows)
            self.explained_variance = float(svd.explained_variance_ratio_.sum())

        return {
            "built": True,
            "n_docs": self.n_docs,
            "dims": dims,
            "explained_variance": round(self.explained_variance, 3),
            "vocab_word": len(word_vec.vocabulary_),
            "vocab_char": len(char_vec.vocabulary_),
        }

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if not (self.word_vec and self.char_vec and self.svd):
            return None
        sparse = hstack([
            self.word_vec.transform(texts), self.char_vec.transform(texts)
        ]).tocsr()
        return l2_normalize(self.svd.transform(sparse))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if not self.ready or not query.strip():
            return []
        with self._lock:
            vec = self._embed([query])
            if vec is None:
                return []
            scores = self.matrix @ vec[0]
            top = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0.02]

    def similar(self, canonical_id: str, k: int = 6) -> list[tuple[str, float]]:
        if not self.ready or canonical_id not in self.ids:
            return []
        with self._lock:
            idx = self.ids.index(canonical_id)
            scores = self.matrix @ self.matrix[idx]
            scores[idx] = -1.0
            top = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0.3]

    def find_duplicates(self, threshold: float = DUP_THRESHOLD) -> list[tuple[str, str, float]]:
        """Near-duplicate pairs, blocked by employer.

        The first version of this compared every posting against every other and
        flagged 143 pairs out of 388 -- because "Site Reliability Engineer" at
        Acme and "Site Reliability Engineer" at Globex share a title and a pile
        of boilerplate, and cosine similarity cannot tell you they are different
        jobs. They are. Different employers, different roles, however similar
        the words.

        So candidates are **blocked on normalised company** and compared only
        within a block. That is standard entity resolution, and it fixes two
        things at once: the precision problem above, and the scaling problem --
        an all-pairs scan is O(n^2), while blocking is O(sum of block^2), which
        for job data is close to linear because no employer has thousands of
        open roles.

        Rows with no employer are excluded rather than lumped together. If we
        cannot say who is hiring, we cannot say two postings are the same job.

        Known limitation, stated rather than hidden: a re-listing whose title
        was substantially *rewritten* ("Backend Engineer" -> "Developer, Core
        Platform") is not caught. Catching it would mean comparing descriptions,
        and within one employer those are near-identical boilerplate -- the
        version that did compare them scored "Head of Marketing" against "Head
        of Design" at 0.99. Between missing some duplicates and inventing them,
        missing is the cheaper error: a flagged duplicate hides a real vacancy.
        """
        if not self.ready or self.matrix.shape[0] < 2:
            return []

        rows = db.query(
            "SELECT canonical_id, company, title FROM jobs WHERE company IS NOT NULL "
            "AND TRIM(company) != ''"
        )
        position = {cid: i for i, cid in enumerate(self.ids)}
        blocks: dict[str, list[int]] = {}
        seniority: dict[int, str | None] = {}
        for r in rows:
            key = norm_company(r["company"])
            idx = position.get(r["canonical_id"])
            if key and idx is not None:
                blocks.setdefault(key, []).append(idx)
                seniority[idx] = label_seniority(r["title"] or "")

        out: list[tuple[str, str, float]] = []
        with self._lock:
            if self.identity_matrix is None:
                return []
            for members in blocks.values():
                if len(members) < 2:
                    continue
                block = self.identity_matrix[members]
                sims = (block @ block.T).toarray()
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        score = float(sims[a, b])
                        if score < threshold:
                            continue
                        # A senior and a non-senior opening at the same employer
                        # with otherwise identical titles are two vacancies, not
                        # one. "Software Engineer" and "Senior Software Engineer"
                        # score 0.88 on titles alone, and collapsing them would
                        # quietly delete a real job from the count.
                        if (seniority.get(members[a]) != seniority.get(members[b])):
                            continue
                        out.append((self.ids[members[a]], self.ids[members[b]],
                                    round(score, 4)))
        return out

    def mark_duplicates(self) -> int:
        """Point each near-duplicate at a single survivor.

        The survivor is the earliest-seen row, so the "first seen" date stays
        truthful. Duplicates are flagged rather than deleted -- the fact that
        one posting appeared on four boards is signal worth keeping.
        """
        pairs = self.find_duplicates()
        if not pairs:
            return 0
        seen = {
            r["canonical_id"]: r["first_seen"]
            for r in db.query("SELECT canonical_id, first_seen FROM jobs")
        }
        marks: list[tuple[str, str]] = []
        for a, b in ((p[0], p[1]) for p in pairs):
            if a not in seen or b not in seen:
                continue
            keeper, dup = (a, b) if seen[a] <= seen[b] else (b, a)
            marks.append((keeper, dup))
        if marks:
            db.executemany(
                "UPDATE jobs SET dup_of = ? WHERE canonical_id = ? AND dup_of IS NULL",
                marks,
            )
        return len(marks)

    def status(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "n_docs": self.n_docs,
            "dims": int(self.matrix.shape[1]) if self.ready else 0,
            "explained_variance": round(self.explained_variance, 3),
            "built_at": self.built_at,
            "age_s": round(time.time() - self.built_at, 1) if self.built_at else None,
            "method": "TF-IDF(word 1-2 + char_wb 3-5) -> TruncatedSVD -> L2, cosine",
        }


index = SemanticIndex()
