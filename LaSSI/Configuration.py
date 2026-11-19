from enum import Enum

# LaSSI v4.5 reads HOnK v2.0.0, kept apart from v4.4's HOnK v1 (Oliver, 16 and 18 Nov 2026: a separate database; the
# shared "honk" table and v4.4's cache/honk_oxstore are never rebuilt). Paths are relative to the working directory.
HONK_GRAPH = "LaSSI/HOnK-v2-0-0.nt"     # the release's N-Triples (a link to it; read in place, never copied)
HONK_CACHE = "cache_v45"                # its RocksDB store, freshness marker and relation adjacency
HONK_TABLE = "honk_v2"                  # the fuzzy matcher's table, in connection.yaml's database (lassi_v45)


class SentenceRepresentation(Enum):
    FullText = 0
    SimpleGraph = 1
    LogicalGraph = 2
    Logical = 3
    SimpleGraphDisabledAPriori = 4
    LogicalGraphDisabledAPriori = 5
    LogicalDisabledAPriori = 6
