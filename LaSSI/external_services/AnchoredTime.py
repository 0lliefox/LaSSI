"""SUTime anchored to a reference date (issue 158, 2026-09-25).

The Java wrapper's `getTimeUnits` runs SUTime inside a CoreNLP pipeline built with `ner.docdate.usePresent = true`,
so a relative expression is resolved against the day the pipeline runs: "Monday" or "five days from Monday" name a
day the text never states, and a different one on every rebuild. When the caller knows the document's own date (a
notice's publication date), relative expressions should be resolved against it.

This module builds a second, time-only pipeline with the wrapper's annotators up to NER and its SUTime options, but
without `usePresent`, and sets each document's DocDate to the reference date. The output has the shape of the
wrapper's (`text` = the value, `type`, `start_char`, `end_char`, `confidence`), with the value formatted by the
wrapper's own `ResolvedTime` hierarchy, so an absolute expression gets exactly the value it got before. It needs the
JVM the wrapper starts (`OldWrapper.getInstance()`); the pipeline is built on first use.
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterable, List, Optional

_PIPELINE = None


def normalise_reference_date(value) -> Optional[str]:
    """`YYYY-MM-DD` for a date, a datetime or an ISO string ("2026-04-13T00:00:00+01:00" -> "2026-04-13"), or None.
    The date is taken as written: a publication time's offset does not move it to another day."""
    if value is None or value == "":
        return None
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return _dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError as e:
        raise ValueError(f"reference_date must be an ISO date or datetime, got {value!r}") from e


def _time_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from StanfordNLPExtractor.OldWrapper import OldWrapper
        OldWrapper.getInstance()                                  # starts the JVM with the wrapper's classpath
        from java.util import Properties
        from edu.stanford.nlp.pipeline import StanfordCoreNLP
        props = Properties()
        props.setProperty("annotators", "tokenize, pos, lemma, ner")  # the wrapper's, up to NER (parse, dcoref follow it)
        props.setProperty("sutime.includeRange", "true")
        props.setProperty("sutime.markTimeRanges", "true")
        _PIPELINE = StanfordCoreNLP(props)
    return _PIPELINE


def time_units(sentences: Iterable[str], reference_date) -> List[list]:
    """One list of time units per sentence, as the wrapper's `getTimeUnits`, resolved against `reference_date`."""
    day = normalise_reference_date(reference_date)
    if day is None:
        raise ValueError("time_units needs a reference date; without one use the wrapper's getTimeUnits")
    pipeline = _time_pipeline()
    from edu.stanford.nlp.pipeline import CoreDocument
    from edu.stanford.nlp.ling import CoreAnnotations
    from edu.stanford.nlp.time import TimeAnnotations
    from org.ufl.hypogator.jackb.disambiguation.dimension.time import ResolvedTime
    out = []
    for sentence in sentences:
        doc = CoreDocument(str(sentence))
        doc.annotation().set(CoreAnnotations.DocDateAnnotation, day)
        pipeline.annotate(doc)
        units = []
        for mention in doc.entityMentions():
            timex = mention.coreMap().get(TimeAnnotations.TimexAnnotation)
            if timex is None:
                continue
            surface = str(mention.text())
            try:
                hierarchy = [str(h) for h in ResolvedTime(timex.value(), mention.text()).toHierarchy()]
            except Exception:                                    # the wrapper's ResolvedTime rejects some values
                hierarchy = []
            offsets = mention.charOffsets()
            units.append({"text": hierarchy[0] if hierarchy else surface, "type": str(timex.timexType()),
                          "start_char": int(offsets.first()), "end_char": int(offsets.second()),
                          "confidence": 1.0})
        out.append(units)
    return out
