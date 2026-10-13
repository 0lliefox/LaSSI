"""The canonical stored shape of a node's property bag.

A bag used to be a `frozenset` of (key, value) pairs, so iteration order followed string hashing and the
pipeline could only reproduce its own output under `PYTHONHASHSEED=0` (see `LaSSI.utils.determinism`). It is
now a **tuple sorted by key**, which is still hashable and still compares equal for equal content, but has a
fixed order that does not depend on the interpreter's hash seed.

Everything that reads a bag either iterates it or wraps it in `dict(...)`, and nothing performs set
operations on one, so a tuple is a drop-in replacement. Keys are sometimes tuples rather than strings, so the
sort is on `repr(key)`: it only has to be total and stable, not meaningful.

Two identical copies of this logic existed before — `create_props_for_singleton` in `ner/node_functions.py`
(and again in `node_functions_X.py`) and `freeze_props` in `ner/structural_rewrites/base.py`. Both now
delegate here.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def _hashable(value: Any) -> Any:
    """Lists are not hashable and appear as both keys and values; tuples of the same items are."""
    return tuple(value) if isinstance(value, list) else value


def freeze_properties(properties: Mapping | Iterable | None) -> tuple:
    """Return *properties* as the canonical stored bag: a tuple of (key, value) pairs sorted by key.

    Accepts a mapping, an iterable of pairs, or an already-frozen bag. An iterable of pairs may repeat a key,
    which a `frozenset` would silently drop only when the whole pair matched; duplicates are resolved here the
    way a dict would, last one wins, so the result is always key-unique.
    """
    if properties is None:
        return ()
    items = properties.items() if hasattr(properties, "items") else properties
    collapsed: dict = {}
    for key, value in items:
        collapsed[_hashable(key)] = _hashable(value)
    return tuple(sorted(collapsed.items(), key=lambda kv: repr(kv[0])))


EMPTY_PROPERTIES: tuple = ()


def is_property_bag(value: Any) -> bool:
    """Is *value* a stored property bag?

    The canonical shape is a tuple of (key, value) pairs. A `frozenset` of pairs is still accepted because
    that was the shape before 2026-09-22 and cached artefacts and third-party callers may still produce one;
    accepting both keeps a type-check from rejecting a bag for being the new shape, which is how the
    conversion first broke the comparison stage.
    """
    if isinstance(value, frozenset):
        return True
    return isinstance(value, tuple) and all(isinstance(kv, tuple) and len(kv) == 2 for kv in value)


def as_mapping(serialised: Any) -> dict:
    """Read a serialised property bag as a mapping, whatever shape it is on disk.

    The canonical on-disk shape is a JSON object. A cache written while the in-memory shape had changed but
    the encoder had not will hold an array of [key, value] pairs instead; both load here, so no reader has to
    care and no cache has to be thrown away.
    """
    if not serialised:
        return {}
    if isinstance(serialised, dict):
        return dict(serialised)
    return {kv[0]: kv[1] for kv in serialised if isinstance(kv, (list, tuple)) and len(kv) == 2}
