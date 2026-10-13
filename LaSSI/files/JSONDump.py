import dataclasses
import json
from collections import defaultdict
from enum import Enum


def isGoodKey(x):
    return x is None or isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def _properties_as_object(d):
    from LaSSI.utils.property_bags import is_property_bag
    props = d.get("properties") if isinstance(d, dict) else None
    if props is None or isinstance(props, dict) or not is_property_bag(props):
        return d
    out = dict(d)
    out["properties"] = dict(props)
    return out


class EnhancedJSONEncoder(json.JSONEncoder):
    """NOTE: `LaSSI/similarities/graph_similarity.py` holds a near-identical copy of this class. This is the
    one `json_dumps` uses, so it is the one that decides the on-disk shape of every cached artefact."""

    def default(self, o):
        if o is None:
            return None
        # A property bag is a tuple of (key, value) pairs in memory (sorted; see LaSSI.utils.property_bags)
        # and was a frozenset of the same pairs before 2026-09-22. json writes a tuple as an array, which
        # would silently change the cache format and break every reader that calls `.items()` on it, so the
        # object shape the format has always had is restored here.
        o = _properties_as_object(o)
        from LaSSI.structures.internal_graph.EntityRelationship import Grouping
        if isinstance(o, defaultdict):
            if not all(map(isGoodKey, o.keys())):
                return {str(k): self.default(v) for k, v in o.items()}
            else:
                return {k: self.default(v) for k, v in o.items()}
        elif isinstance(o, dict):
            if not all(map(isGoodKey, o.keys())):
                return {str(k): self.default(v) for k, v in o.items()}
            else:
                return {k: self.default(v) for k, v in o.items()}
        elif isinstance(o, str) or isinstance(o, int) or isinstance(o, float) or isinstance(o, dict):
            return o
        elif isinstance(o, list) or isinstance(o, set) or isinstance(o, tuple):
            L = [self.default(x) for x in o]
            try:
                return super().default(L)
            except:
                return L
        elif dataclasses.is_dataclass(o):
            return self.default(dataclasses.asdict(o))
        elif isinstance(o, frozenset):
            return dict(o)
        elif isinstance(o, Enum):
            return o.value
        return super().default(o)

def obj_pickle(obj):
    import pickle
    return pickle.dumps(obj)

def obj_unmarshall(file):
    import pickle
    data = pickle.load(file)
    # close the file
    file.close()
    return data

def json_dumps(obj):
    return json.dumps(obj, cls=EnhancedJSONEncoder, indent=4)
