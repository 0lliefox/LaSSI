import dataclasses
import json
from collections import defaultdict
from enum import Enum


def isGoodKey(x):
    return x is None or isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if o is None:
            return None
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
