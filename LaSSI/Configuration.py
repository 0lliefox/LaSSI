from enum import Enum

# LaSSI v4.5 reads HOnK v1.6.1, the errors-only release LaSSI freezes on (Oliver, 20 Nov 2026: "Errors-only for both"),
# kept apart from v4.4's HOnK v1.5.0 and from HOnK v2.0.0, which v4.5 read first (Oliver, 16 and 18 Nov: a separate
# database; the shared "honk" table and v4.4's cache/honk_oxstore are never rebuilt; 20 Nov: the table is honk_v161).
# v2.0.0's names were LaSSI/HOnK-v2-0-0.nt, cache_v45 and honk_v2; a store or table built from it is never reused.
# Paths are relative to the working directory.
HONK_GRAPH = "LaSSI/HOnK-v1-6-1.nt"     # the release's N-Triples (a link to it; read in place, never copied)
HONK_CACHE = "cache_v161"               # its RocksDB store, freshness marker and relation adjacency
HONK_TABLE = "honk_v161"                # the fuzzy matcher's table, in connection.yaml's database (lassi_v45)


class SentenceRepresentation(Enum):
    FullText = 0
    SimpleGraph = 1
    LogicalGraph = 2
    Logical = 3
    SimpleGraphDisabledAPriori = 4
    LogicalGraphDisabledAPriori = 5
    LogicalDisabledAPriori = 6
