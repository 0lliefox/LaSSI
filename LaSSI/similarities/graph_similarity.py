__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"
import dataclasses
import functools
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from nltk.stem.wordnet import WordNetLemmatizer

from LaSSI.similarities.string_similarity_factory import StringSimilarity
from LaSSI.structures.internal_graph.EntityRelationship import Relationship, Grouping, Singleton, NodeEntryPoint
from LaSSI.utils.configurations import LegacySemanticConfiguration

# from gsmtosimilarity.string_similarity_factory import StringSimilarity
# from inference_engine.EntityRelationship import Relationship, Grouping, Singleton, NodeEntryPoint
# from logical_repr.Sentences import Formula

negations = {'not', 'no', 'but'}
location_types = {'GPE', 'LOC'}
lemmatizer = WordNetLemmatizer()

@dataclass(order=True, frozen=True, eq=True)
class Graph:
    edges: List[Relationship]  # A graph is defined as a collection of edges

def isGoodKey(x):
    return x is None or isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if o is None:
            return None
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
        elif isinstance(o, Grouping):
            return o.name
        return super().default(o)


class SimilarityScore:  # Defining the graph similarity score
    def __init__(self, cfg:LegacySemanticConfiguration):
        self.entity_similarity = StringSimilarity(cfg, 'string_similarity')
        # if 'verb_similarity' in cfg:
        self.rel_similarity = StringSimilarity(cfg, 'verb_similarity')
        # else:
        #     self.rel_similarity = self.entity_similarity

    @functools.lru_cache
    def string_distance(self, x: str, y: str) -> float:  # between 0 and 1
        return 1.0 - self.entity_similarity.string_similarity(x, y)

    @functools.lru_cache
    def string_similarity(self, x: str, y: str) -> float:  # between 0 and 1
        return self.entity_similarity.string_similarity(x, y)

    @functools.lru_cache
    def properties_distance(self, x: frozenset, y: frozenset) -> float:
        if (len(x) == 0) and (len(y) == 0):
            return 0
        x = dict(x)
        y = dict(y)
        L = set(x.keys())
        I = L.intersection(set(y.keys()))
        U = L.union(set(y.keys()))
        missing = len(U.difference(I))
        others = 0
        other_dist = 0
        for k in I:
            l = x[k]
            r = y[k]
            if type(l) != type(r):
                missing = missing + 1.0
            else:
                others = others + 1.0
                compute = 0.0
                number = False
                if isinstance(l, str):
                    compute = self.string_distance(l, r)
                elif isinstance(l, bool):
                    l = 1 if l else 0
                    r = 1 if r else 0
                    number = True
                else:
                    number = True
                if number:
                    compute = (abs(l - r) * 1.0) / (1.0 + abs(l - r) * 1.0)
                other_dist = other_dist + compute
        return (missing + other_dist) / (missing + others)

    @functools.lru_cache
    def singleton_dist(self, x: Singleton, y: Singleton, isRel=False) -> float:  # between 0 and 1
        if (x is None) or (y is None):
            return 0.0
        elif isRel:
            return 1.0 - self.rel_similarity.string_similarity(x.named_entity, y.named_entity)
        else:
            return self.string_distance(x.named_entity, y.named_entity)

    @functools.lru_cache
    def entity_distance(self, lhs: NodeEntryPoint, rhs: NodeEntryPoint) -> float:
        if (lhs is None) or (rhs is None):
            return 0.0
        elif isinstance(lhs, Singleton) and isinstance(rhs, Singleton):
            return self.singleton_dist(lhs, rhs)
        elif isinstance(lhs, Singleton):
            if rhs.type == Grouping.AND:
                return 1.0
            elif rhs.type == Grouping.OR:
                return self.entity_distance(lhs, min(rhs.entities, key=lambda z: self.entity_distance(lhs, z)))
            elif rhs.type == Grouping.NOT:
                assert len(rhs.entities) == 1
                return 1 - self.entity_distance(lhs, rhs.entities[0])
            elif rhs.type == Grouping.NEITHER:
                return 0
            else:
                raise ValueError(str(rhs.type) + " is not supported")
        elif isinstance(rhs, Singleton):
            if lhs.type == Grouping.AND:
                return self.entity_distance(min(lhs.entities, key=lambda z: self.entity_distance(z, rhs)), rhs)
            elif lhs.type == Grouping.OR:
                return 1.0
            elif lhs.type == Grouping.NOT:
                assert len(lhs.entities) == 1
                return 1 - self.entity_distance(lhs.entities[0], rhs)
            elif lhs.type == Grouping.NEITHER:
                return 1 - self.entity_distance(min(lhs.entities, key=lambda z: self.entity_distance(z, rhs)), rhs)
            else:
                raise ValueError(str(rhs.type) + " is not supported")
        else:
            assert len(lhs.entities) > 0
            assert len(rhs.entities) > 0
            S1 = set(lhs.entities)
            S2 = set(rhs.entities)
            d = defaultdict(set)
            d_inv = defaultdict(set)
            matches = []
            total_cost = 0

            for z in S1:
                yMin = min(S2, key=lambda k: self.entity_distance(z, k))
                distance = self.entity_distance(z, yMin)
                total_cost = total_cost + distance
                matches.append(tuple([z, distance, yMin]))
                d[z].add(yMin)
                d_inv[yMin].add(z)
            S1_inv = set()
            S2_inv = set()
            empty_set = set()

            for edge2 in rhs.entities:
                S2_inv = S2_inv.union(d_inv.get(edge2, empty_set))
            for edge1 in lhs.entities:
                S1_inv = S1_inv.union(d.get(edge1, empty_set))
            # unmatchingDistance = (len(S1.difference(S2_inv)) + len(S2.difference(S1_inv)))
            if lhs.type == Grouping.AND:
                if rhs.type == Grouping.AND:
                    if len(S2) == len(S1_inv):
                        return float(total_cost) / float(len(matches))
                    else:
                        return 1.0  # If some of elements could not be derived from the left, the left does not entail the right
                elif rhs.type == Grouping.OR:
                    return float(total_cost) / float(len(matches))  # Assuming there is at least one alignment
                else:
                    return 1.0  # Contradiction
            elif lhs.type == Grouping.OR:
                if rhs.type == Grouping.AND:
                    return 1.0  # Cannot convert a disjunction into a conjunction
                elif rhs.type == Grouping.OR:
                    return float(total_cost + len(S2.difference(S1_inv))) / float(
                        len(matches) + len(S2.difference(S1_inv)))
                else:
                    return 1.0  # Contradiction
            elif lhs.type == Grouping.NOT:
                if rhs.type == Grouping.NOT:
                    return self.entity_distance(lhs.entities[0], rhs.entities[0])
                else:
                    return 1.0
            elif lhs.type == Grouping.NEITHER:
                if rhs.type == Grouping.NEITHER:
                    if len(S2) == len(S1_inv):
                        return float(total_cost) / float(len(matches))
                    else:
                        return 1.0  # If some of elements could not be derived from the left, the left does not entail the right
                else:
                    return 1.0  # Contradiction

    @functools.lru_cache
    def edge_distance(self, x: Relationship, y: Relationship):
        # L = (x.source.named_entity,x.edgeLabel.named_entity,x.target.named_entity)
        # R = (y.source.named_entity,y.edgeLabel.named_entity,y.target.named_entity)
        # print (str(L) +" vs "+str(R))
        srcSim = 1.0 - self.entity_distance(x.source, y.source)
        dstSim = 1.0 - self.entity_distance(x.target, y.target)
        edgeSim = 1.0 - self.singleton_dist(x.edgeLabel, y.edgeLabel, True)
        edgeSim = edgeSim * (1 if x.isNegated == y.isNegated else 0)
        return 1.0 - (srcSim * dstSim * edgeSim)

    def graph_distance(self, g1: Graph, g2: Graph)->float:
        if len(g1.edges) == 0 or len(g2.edges) == 0:
            return []
        d = defaultdict(set)
        d_inv = defaultdict(set)
        S1 = set(g1.edges)
        S2 = set(g2.edges)
        total_cost = 0
        count = 0
        L = []
        for edge1 in g1.edges:
            edge2_cand = min(g2.edges, key=lambda edge2: self.edge_distance(edge1, edge2))
            distance = self.edge_distance(edge1, edge2_cand)
            L.append(tuple([edge1, distance, edge2_cand]))
            total_cost += distance
            count += 1
            d[edge1].add(edge2_cand)
            d_inv[edge2_cand].add(edge1)
        S1_inv = set()
        S2_inv = set()
        empty_set = set()
        for edge2 in g2.edges:
            S2_inv = S2_inv.union(d_inv.get(edge2, empty_set))
        for edge1 in g1.edges:
            S1_inv = S1_inv.union(d.get(edge1, empty_set))
        unmatchingDistance = (len(S1.difference(S2_inv)) + len(S2.difference(S1_inv)))
        unmatchingSimilarity = 1.0 - (unmatchingDistance / (1 + unmatchingDistance))
        totalSimilarity = 1.0 - (total_cost / count)
        # total_cost += ((len(S1.difference(S2_inv)) + len(S2.difference(S1_inv))) / 2.0)
        return 1.0 - totalSimilarity * unmatchingSimilarity


def read_graph_from_file(result_json):
    parsed_json = {}
    with (open(result_json) as user_file):  # result.json from C++
        parsed_json = json.load(user_file)
    return parsed_json


# def load_file_for_similarity(result_json, stanza_row, rejected_edges, non_verbs, do_rewrite, simplistic):
#     with (open(result_json) as user_file):  # result.json from C++
#         parsed_json = json.load(user_file)
#         from graph_repr.internal_graph import to_internal_graph
#         g, a, b = to_internal_graph(parsed_json, stanza_row, rejected_edges, non_verbs, do_rewrite, simplistic)
#         return g
#
#
# if __name__ == '__main__':
#     # TODO: Load all graph results from output of GSM C++ code
#     result_json = './result.json'
#     print(load_file_for_similarity(result_json))
