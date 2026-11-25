__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

import re

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule, is_date_like
from LaSSI.structures.internal_graph.EntityRelationship import Singleton
from LaSSI.utils.datetime_canon import canonicalize_datetime_string


# Date/time typing goes through base.is_date_like — the local copy this
# replaced spelled the NER source "SUTime" unuppercased, so it silently never
# matched a SUTIME-typed node.
_is_date = is_date_like

# A clock time a sentence states in UTC: "5pm UTC", "11:30am GMT", "noon UTC".
_UTC_CLOCK_RE = re.compile(
    r"^\s*(?:(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<ap>am|pm)|(?P<word>noon|midday|midnight))\s+(?:utc|gmt)\s*$",
    re.IGNORECASE,
)


def _utc_clock(name):
    """The (hour, minute) of a clock time stated in UTC (``"5pm UTC"`` -> (17, 0)), or ``None``."""
    if not isinstance(name, str):
        return None
    m = _UTC_CLOCK_RE.match(name.replace("<dot>", "."))
    if m is None:
        return None
    if m.group("word"):
        return (0, 0) if m.group("word").lower() == "midnight" else (12, 0)
    h, mi = int(m.group("h")), int(m.group("m") or 0)
    if not (1 <= h <= 12 and 0 <= mi <= 59):
        return None
    return (h % 12 + (12 if m.group("ap").lower() == "pm" else 0), mi)


def _fold_utc_clock_times(items):
    """Policy v3.1: a claim's time is read in the zone it states. For "at 5pm UTC on 14 April 2026", SUTime folds
    the clock time into the date's ISO value (2026-04-14T17:00:00Z) but leaves "5pm UTC" as a TIME entry of its own,
    which the dedup by ISO string can't merge. When an ISO entry in the same list holds that UTC clock time, the two
    are the same instant: the clock entry's properties the ISO entry lacks (its case marker, e.g. ``14:at``) move
    onto the ISO entry, and the clock entry is dropped. A clock time no ISO entry holds is kept as it is. The list
    stays a list, as the dedup leaves it when two entries merge into one: every kernel property is iterated as a list
    downstream (KernelPostProcessor.remove_duplicate_properties)."""
    iso = [k for k, it in enumerate(items)
           if isinstance(it, Singleton) and _is_date(it)
           and canonicalize_datetime_string(getattr(it, "named_entity", None)) == it.named_entity]
    out, drop = list(items), set()
    for i, it in enumerate(items):
        if not (isinstance(it, Singleton) and _is_date(it)):
            continue
        hm = _utc_clock(getattr(it, "named_entity", None))
        if hm is None:
            continue
        same = [k for k in iso if out[k].named_entity[11:16] == f"{hm[0]:02d}:{hm[1]:02d}"]
        if not same:
            continue
        k = same[0]
        props = dict(out[k].properties)
        for key, val in dict(it.properties).items():
            props.setdefault(key, val)
        out[k] = out[k].update_node_props(props)
        drop.add(i)
    return [it for i, it in enumerate(out) if i not in drop]


def _canonicalize_singleton_datetime(item):
    """If *item* is a DATE/TIME singleton whose ``named_entity`` parses as a
    datetime, return a copy whose ``named_entity`` is the canonical ISO 8601
    UTC string (e.g. ``2026-04-14T14:00:00Z``).  Otherwise return *item* as-is.

    Handles both surface forms emitted upstream:
      * ``2026-04-14T14:00Z`` (lower- or upper-cased) from the ISO 8601 regex
        path in ``ResolveSingleSentence``.
      * ``2026-04-14T14-0`` from the SUTime path (colons replaced by ``-``,
        minutes truncated).
    Both collapse to the same canonical form, so downstream comparisons
    succeed via simple string equality without needing any datetime-aware
    fallback in the comparison layer.
    """
    if not _is_date(item):
        return item
    canonical = canonicalize_datetime_string(getattr(item, "named_entity", None))
    if canonical is None or canonical == item.named_entity:
        return item
    return item.update_name(canonical)


class TimeCanonicalisationRule(StructuralRewriteRule):
    """Canonicalise TIME property values on the kernel.

    Four cleanups, all driven by the fact that SUTime/CoreNLP collapses
    surface mentions like "8pm" and "14 April 2026" into the same ISO 8601
    string (e.g. ``2026-04-14T20-0``).  Anything those mentions added that
    is already encoded in the ISO form is redundant noise:

    * **Canonicalise the ISO string itself** to ``YYYY-MM-DDTHH:MM:SSZ`` so
      that two pipelines emitting the same instant in slightly different
      surface forms (``2026-04-14T14:00Z`` from the ISO regex, ``2026-04-14T14-0``
      from SUTime) reach downstream code as the SAME string.  Without this
      pass, the ex-post comparison layer carries a runtime fallback that has
      to re-parse and re-canonicalise on every cell — wasteful and easy to
      miss in new code paths.

    * **Strip `nummod`** on DATE/TIME/SUTime-typed entries.  The bare number
      that produced the entry (`8` for "8pm", `2026` for the year) lives
      inside the ISO string already, and as a standalone property reads as
      "nummod:8" which doesn't obviously mean "8pm".

    * **Deduplicate by `named_entity`**.  When multiple TIME entries share
      the same ISO form, keep the one with the richer case-marker /
      position bag (more positional keys like ``18:at``, plus ``type``),
      falling back to the longer character span.  The other entry is
      typically the year-only duplicate ("2026") which carries no useful
      case info.  Canonicalising the named_entity FIRST is essential — two
      entries that only differed by surface form (``T14:00Z`` vs ``T14-0``)
      otherwise stayed as separate dictionary keys here and survived dedup.

    * **Fold a clock time stated in UTC** ("5pm UTC") into the ISO entry that
      holds that UTC clock time (policy v3.1; see ``_fold_utc_clock_times``)."""

    name = "time_canonicalisation"
    phase = "post_logical_rewrite"

    @staticmethod
    def _score(node):
        if not hasattr(node, 'properties'):
            return (0, 0)
        props = dict(node.properties)
        info_keys = 0
        for k in props:
            try:
                float(k)
                info_keys += 1
            except (TypeError, ValueError):
                if k == 'type':
                    info_keys += 1
        span = (getattr(node, 'max', 0) or 0) - (getattr(node, 'min', 0) or 0)
        return (info_keys, span)

    @classmethod
    def _pick_richer(cls, a, b):
        return a if cls._score(a) >= cls._score(b) else b

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton):
            return None
        props = dict(kernel.properties) if kernel.properties else {}
        time_value = props.get('TIME')
        if time_value is None:
            return None
        time_list = list(time_value) if isinstance(time_value, (list, tuple)) else [time_value]
        # Match if any TIME entry has a strippable nummod, a non-canonical
        # `named_entity` surface form, or there are multiple entries
        # (duplicates to merge).
        if len(time_list) > 1:
            return {"time_list": time_list}
        for item in time_list:
            if not isinstance(item, Singleton) or not _is_date(item):
                continue
            if 'nummod' in dict(item.properties):
                return {"time_list": time_list}
            # Non-canonical surface form (e.g. "2026-04-14T14-0" or
            # "2026-04-14T14:00Z" instead of the canonical
            # "2026-04-14T14:00:00Z") also triggers the rule so the
            # construction-time normaliser has a chance to rewrite the
            # named_entity even when there's nothing else to clean up.
            canonical = canonicalize_datetime_string(item.named_entity)
            if canonical is not None and canonical != item.named_entity:
                return {"time_list": time_list}
        return None

    def apply(self, kernel, bindings, ctx):
        time_list = bindings["time_list"]
        # First pass: canonicalise the named_entity of each DATE/TIME entry to
        # ISO 8601 UTC, then strip nummod. Canonicalisation has to happen
        # BEFORE the dedup pass below — otherwise two entries that describe
        # the same instant in different surface forms (ISO-with-Z vs SUTime
        # "T14-0") survive as separate keys in `by_name`.
        cleaned = []
        for item in time_list:
            if isinstance(item, Singleton) and _is_date(item):
                item = _canonicalize_singleton_datetime(item)
                props = dict(item.properties)
                if 'nummod' in props:
                    props = {k: v for k, v in props.items() if k != 'nummod'}
                    item = item.update_node_props(props)
            cleaned.append(item)
        # A clock time stated in UTC that an ISO entry already holds is that entry (policy v3.1).
        cleaned = _fold_utc_clock_times(cleaned)
        # Second pass: dedupe by named_entity, keeping the richer entry.
        by_name = {}
        for item in cleaned:
            key = getattr(item, 'named_entity', None) or id(item)
            existing = by_name.get(key)
            by_name[key] = item if existing is None else self._pick_richer(existing, item)
        deduped = list(by_name.values())
        props = dict(kernel.properties) if kernel.properties else {}
        props = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in props.items()}
        # Mirror the shape of the original TIME value (bare singleton vs list)
        # rather than always coercing to a list — downstream code branches on
        # `isinstance(value, (list, tuple))` and corrupting a previously bare
        # TIME entry into a list trips up the FOL expansion lookup keys.
        original = props.get('TIME')
        if isinstance(original, (list, tuple)) or len(deduped) > 1:
            props['TIME'] = deduped
        else:
            props['TIME'] = deduped[0]
        return kernel.update_node_props(props)
