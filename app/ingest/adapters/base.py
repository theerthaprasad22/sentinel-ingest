"""Adapter contract.

A source is not one way of getting data -- it is an ordered list of them. That
is the whole point of `Strategy`: when the JSON API starts returning 403, the
scheduler walks down to the RSS feed, and when that breaks it walks down to
parsing the HTML listing. Each rung is slower and uglier than the one above it,
and each one buys another week before a human has to care.

Parsers return loose dicts. Normalisation, identity and validation all happen
downstream in `normalize.py`, so an adapter only ever has to answer one
question: where are the fields on *this* source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from lxml import etree, html as lxml_html


@dataclass(frozen=True)
class Strategy:
    """One way to pull a source."""

    name: str                                   # "api" | "rss" | "html"
    url: str
    expected: str                               # "json" | "xml" | "html"
    parse: Callable[[str], list[dict]]
    # Cost is a rough "how much do we dislike using this" -- the scheduler
    # prefers cheaper rungs and only descends under failure.
    cost: int = 1
    navigation: bool = False                    # send browser navigation headers?


@dataclass(frozen=True)
class Adapter:
    name: str
    kind: str
    strategies: tuple[Strategy, ...]
    cadence_s: float = 300.0
    # One-line statement of why we are allowed to be here. Rendered in the UI
    # next to the source, because "is this source fair game" should be visible,
    # not buried in a design doc.
    licence_note: str = ""
    homepage: str = ""

    def strategy(self, name: str) -> Strategy | None:
        return next((s for s in self.strategies if s.name == name), None)


# --------------------------------------------------------------------------
# Resilient extraction helpers
# --------------------------------------------------------------------------

def first_text(node, xpaths: list[str]) -> tuple[str, int]:
    """Try each XPath in order; return (text, index_of_the_one_that_matched).

    Candidate lists rather than a single selector are the cheap half of markup
    resilience: boards rename `.job-title` to `.posting-title` far more often
    than they stop putting the title in an `<h2>` or a `data-` attribute.

    Returning *which* candidate won is the part that matters. When candidate 0
    stops matching across a whole batch, that is the drift signal, and we want
    it visible before the fill-rate collapses and rows start going missing.
    Index -1 means nothing matched.
    """
    for idx, xp in enumerate(xpaths):
        try:
            found = node.xpath(xp)
        except (etree.XPathEvalError, TypeError):
            continue
        for item in found:
            text = item if isinstance(item, str) else "".join(item.itertext())
            text = " ".join(str(text).split())
            if text:
                return text, idx
    return "", -1


def extraction_confidence(matched: dict[str, int]) -> float:
    """Fraction of fields that matched their *first* candidate.

    1.0 means the page looks exactly as expected. 0.4 means we are limping
    along on fallback selectors and someone should refresh them before the last
    one goes too. This number is written to every row, so a query can find the
    batch where extraction started degrading.
    """
    if not matched:
        return 0.0
    primary = sum(1 for v in matched.values() if v == 0)
    return round(primary / len(matched), 3)


def parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_xml_items(text: str, item_tag: str = "item") -> list:
    """RSS/Atom entries. Namespace-agnostic on purpose -- half the job feeds in
    the wild declare a namespace they then do not use consistently."""
    try:
        # recover=True: feeds routinely contain unescaped ampersands, and
        # refusing to parse the whole feed over one bad character is a
        # self-inflicted outage.
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
        root = etree.fromstring(text.encode("utf-8"), parser=parser)
    except (etree.XMLSyntaxError, ValueError):
        return []
    if root is None:
        return []
    return root.xpath(f"//*[local-name()='{item_tag}'] | //*[local-name()='entry']")


def xml_field(node, names: list[str]) -> str:
    for n in names:
        found = node.xpath(f"./*[local-name()='{n}']")
        if found:
            text = "".join(found[0].itertext()).strip()
            if not text:
                text = (found[0].get("href") or "").strip()
            if text:
                return text
    return ""


def parse_html(text: str):
    return lxml_html.fromstring(text)
