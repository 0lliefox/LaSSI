"""LaSSI v4.5: LaSSI v4.4 reading HOnK v2.0.0 (Oliver, 18-19 Nov 2026).

Each test pins one way v4.5 reads HOnK v2 differently from v4.4 (gap list: honk_v170/review_2026-11-16/
lassi_v45_gap_list.md) and fails on v4.4 (4f70287). Hand-built inputs only: no parser, no Postgres, no HOnK release,
no CURB text.
"""
import unittest
from unittest import mock

from LaSSI.HOnK.HOnK import HOnK


class FacIsAPlaceTypeOfItsOwn(unittest.TestCase):
    """HOnK v2 types 3,003,646 facilities FAC (stations, buildings, streets). v4.4 had no FAC branch, so they lost
    their place type ("None"). Oliver, 19 Nov: FAC is a type of its own, ranked GPE > LOC > FAC."""

    def test_fac_alone_is_fac(self):
        self.assertEqual(HOnK.most_specific_type(["FAC"]), "FAC")
        self.assertEqual(HOnK.most_specific_type(["FAC", "Noun"]), "FAC")

    def test_fac_ranks_below_gpe_and_loc(self):
        # Newcastle is GPE, LOC and FAC in v2 and stays GPE; a name that is LOC and FAC stays LOC, as in v4.4.
        self.assertEqual(HOnK.most_specific_type(["FAC", "GPE", "LOC"]), "GPE")
        self.assertEqual(HOnK.most_specific_type(["FAC", "LOC"]), "LOC")

    def test_fac_becomes_space_in_logical_forms(self):
        from LaSSI.structures.extended_fol.rewrite_kernels import type_conversion
        self.assertEqual(type_conversion.get("FAC"), "SPACE")

    def test_lower_case_word_never_takes_a_capitalised_facility(self):
        # "station" in lower case must not become the facility "Station" (FAC), as it never becomes a GPE/LOC "Nice".
        from LaSSI.ner import ResolveMultiEntity as RME

        class Table:
            def typedFuzzyMatch(self, threshold, term):
                return {1.0: [("Station", "FAC")]} if term == "station" else {}

        with mock.patch.object(RME, "lemmatize_verb", lambda w: w):
            found = RME.process_sentence(([("station", 0, 7)], Table(), 0.8, 0.8, "HOnK", HOnK, 1.0, None))
        self.assertEqual(found, [])

    def test_capitalised_name_keeps_its_facility_type(self):
        from LaSSI.ner import ResolveMultiEntity as RME

        class Table:
            def typedFuzzyMatch(self, threshold, term):
                return {1.0: [("Haymarket", "FAC")]} if term == "haymarket" else {}

        with mock.patch.object(RME, "lemmatize_verb", lambda w: w):
            found = RME.process_sentence(([("Haymarket", 0, 9)], Table(), 0.8, 0.8, "HOnK", HOnK, 1.0, None))
        self.assertIn("FAC", {e.type for e in found})


class LocationNounIsALocation(unittest.TestCase):
    """HOnK v2 types Parmenides' "city centre", "city center", "city", "station" LocationNoun, no longer LOC, so v4.4's
    table would type them noun only (1,213 labels v4.4 typed LOC). Oliver, 19 Nov: LaSSI's type mapping reads the
    class LocationNoun as a location."""

    def test_location_noun_is_loc(self):
        self.assertEqual(HOnK.most_specific_type(["LocationNoun"]), "LOC")

    def test_city_centre_types_as_in_v44(self):
        # "city centre" in v2: Concept, CountableNoun, LocationNoun, Noun. v4.4's table gave it noun,LOC.
        types = ["Concept", "CountableNoun", "LocationNoun", "Noun"]
        generalised = {HOnK.most_specific_type([t]) for t in types} - {"None"}
        self.assertEqual(generalised, {"LOC", "noun"})

    def test_only_the_class_itself(self):
        # FacilityNoun, RouteNoun, AccessPointNoun (bridge, platform, bus station) had no LOC in v4.4 and get none.
        for cls in ("FacilityNoun", "RouteNoun", "AccessPointNoun", "GeoSuffixNoun"):
            self.assertEqual(HOnK.most_specific_type([cls]), "noun", cls)

    def test_a_named_place_still_ranks_first(self):
        self.assertEqual(HOnK.most_specific_type(["GPE", "LocationNoun"]), "GPE")
        self.assertEqual(HOnK.most_specific_type(["FAC", "LocationNoun"]), "LOC")


def _v1_graph(cache_dir):
    """A HOnK in memory holding HOnK v1's shape for two statements: "fast neqTo1 slow" and "player partOf1 team", each
    numbered relation an instance of its relation (neqTo1 a neqTo), as HOnK v1.5.0 states them."""
    import pyoxigraph as ox
    h = HOnK(cache_dir, "u", "p", "localhost", 0, False)
    h.start()
    H = "https://ofox.co.uk/honk#"
    RDF_TYPE = ox.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    LABEL = ox.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    n = lambda local: ox.NamedNode(H + local)
    add = lambda s, p, o: h.graph.add(ox.Quad(s, p, o))
    for inst, rel, s, o in (("neqTo1", "neqTo", "fast", "slow"), ("partOf1", "partOf", "player", "team")):
        add(n(inst), RDF_TYPE, n(rel))
        add(n(s), n(inst), n(o))
        for w in (s, o):
            add(n(w), LABEL, ox.Literal(w))
    return h


class AdjacencyIsBuiltWithoutHoldingTheGraph(unittest.TestCase):
    """v4.4 built the relation adjacency by keeping every edge and every end's IRI, then scanning every rdfs:label in
    the store; on HOnK v2 (13.7M partOf statements, 24M labels) that is several GB before the tables exist. v4.5 looks
    each end's labels up as the edges stream; the tables are the same."""

    class _NoLabelScan:
        """A store that refuses a scan of every rdfs:label."""
        LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

        def __init__(self, store):
            self._store = store

        def quads_for_pattern(self, s, p, o, g=None):
            if s is None and p is not None and p.value == self.LABEL:
                raise AssertionError("scanned every rdfs:label")
            return self._store.quads_for_pattern(s, p, o, g)

        def __getattr__(self, name):
            return getattr(self._store, name)

    def test_tables_built_without_a_label_scan(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            h = _v1_graph(tmp)
            try:
                h.graph = self._NoLabelScan(h.graph)
                h._build_relation_adjacency()
                self.assertEqual(h._neqTo_adj, {"fast": {"slow"}, "slow": {"fast"}})
                self.assertEqual(h._partOf_wholes, {"player": {"team"}})
            finally:
                h.graph = h.graph._store
                h.stop()


class NTriplesHOnKIsLoadedInChunks(unittest.TestCase):
    """v4.4 loaded HOnK with Store.load, which holds the whole file in memory (on a tenth of HOnK v2.0.0: 8.5 GB; the
    release is 15.4 GB of N-Triples). v4.5 bulk-loads an N-Triples HOnK a chunk of lines at a time."""

    LINES = [
        '_:r <https://ofox.co.uk/honk/logic#logicalConstructName> "space" .',
        '<https://ofox.co.uk/honk/id/fast> <http://www.w3.org/2000/01/rdf-schema#label> "fast" .',
        '<https://ofox.co.uk/honk/id/slow> <http://www.w3.org/2000/01/rdf-schema#label> "slow" .',
        '<https://ofox.co.uk/honk#oppositeOf1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://ofox.co.uk/honk#oppositeOf> .',
        '<https://ofox.co.uk/honk/id/fast> <https://ofox.co.uk/honk#oppositeOf1> <https://ofox.co.uk/honk/id/slow> .',
        '_:r <https://ofox.co.uk/honk/logic#logicalConstructProperty> "near place" .',
        '<https://ofox.co.uk/honk#rule> <https://ofox.co.uk/honk/logic#classification> _:r .',
    ]

    def test_store_is_built_without_store_load(self):
        import os, tempfile
        import pyoxigraph as ox
        from FunctionalMatch.rdf.RDFGraph import RDFGraph
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "honk.nt")
            with open(path, "w") as f:
                f.write("\n".join(self.LINES) + "\n")
            h = HOnK(os.path.join(tmp, "cache"), "u", "p", "localhost", 0, False)
            # Store.load (RDFGraph.parse) must not be used; chunks of 2 lines split the blank node's triples apart.
            with mock.patch.object(RDFGraph, "parse", side_effect=AssertionError("Store.load holds the whole file")), \
                    mock.patch.object(HOnK, "_BULK_LOAD_LINES", 2):
                h.start(path)
            try:
                self.assertEqual(len(h.graph), len(self.LINES))
                blank = {q.subject for q in h.graph if isinstance(q.subject, ox.BlankNode)}
                self.assertEqual(len(blank), 1)                      # one blank node, as in the file
                (node,) = blank
                self.assertEqual(len(list(h.graph.quads_for_pattern(None, None, node, None))), 1)
                self.assertTrue(os.path.exists(os.path.join(tmp, "cache", "honk_oxstore.mtime")))
            finally:
                h.stop()


class HonkTableIsStreamedFromTheGraphFile(unittest.TestCase):
    """v4.4 built the honk table with one SPARQL query over every typed node and a dict of every label (on HOnK v2.0.0:
    20M type and 24M label triples at once). v4.5 streams the N-Triples file and sorts on disk; each label's types
    are the same."""

    H, ID = "https://ofox.co.uk/honk#", "https://ofox.co.uk/honk/id/"
    NODES = {  # node: (label, types)
        "city_centre": ("city centre", ["Concept", "CountableNoun", "LocationNoun", "Noun"]),
        "Newcastle--2641673": ("Newcastle", ["GPE"]),
        "Newcastle--x": ("Newcastle", ["FAC"]),
        "Newcastle": ("Newcastle", ["Noun", "ProperNoun"]),
        "Haymarket--parmenides-locations": ("Haymarket", ["FAC"]),
        "back--slash": ("back\\slash", ["Noun"]),
        "run": ("run", ["Verb", "Noun", "VB"]),
        "walk": ("walk", ["Verb", "Noun"]),
        "untyped": ("untyped", []),
    }

    def _graph(self, tmp):
        import os
        lines = []
        for node, (label, types) in self.NODES.items():
            iri = f"<{self.ID}{node}>"
            esc = label.replace("\\", "\\\\")
            lines.append(f'{iri} <http://www.w3.org/2000/01/rdf-schema#label> "{esc}" .')
            lines += [f"{iri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{self.H}{t}> ." for t in types]
        path = os.path.join(tmp, "honk.nt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    @staticmethod
    def _rows(text):
        rows = {}
        for line in text.splitlines()[1:]:
            _, idx, t, types = line.split("\t")
            assert idx == t
            rows[idx] = types
        return rows

    def test_table_rows_without_the_store_query(self):
        import io, os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            h = HOnK(os.path.join(tmp, "cache"), "u", "p", "localhost", 0, False)
            h.start(self._graph(tmp))
            try:
                # v4.4's rule over the store, for comparison
                expected = {}
                for label, types in h.getTypedObjects().items():
                    gen = {HOnK.most_specific_type([t]) for t in types}
                    if len(gen) > 1:
                        gen.discard("None")
                    expected[label.replace("\\", "\\\\")] = gen
                with mock.patch.object(HOnK, "getTypedObjects", side_effect=AssertionError("every typed node at once")):
                    out = io.StringIO()
                    h.dumpTypedObjectsToTAB(out)
            finally:
                h.stop()
        rows = self._rows(out.getvalue())
        self.assertEqual({k: set(v.split(",")) for k, v in rows.items()}, expected)
        self.assertEqual(set(rows["Newcastle"].split(",")), {"GPE", "FAC", "noun"})
        self.assertEqual(set(rows["city centre"].split(",")), {"LOC", "noun"})

    def test_type_order_follows_v44s_table(self):
        import os, tempfile
        from LaSSI.HOnK import honk_table
        with tempfile.TemporaryDirectory() as tmp:
            graph = self._graph(tmp)
            ref = os.path.join(tmp, "v44.tsv")
            with open(ref, "w") as f:
                # Haymarket's set changed (v4.4: noun,LOC; here FAC), run's set unchanged (kept as v4.4 wrote it),
                # and {verb, noun} is most often "verb,noun" in v4.4's table.
                f.write("run\tnoun,verb\nHaymarket\tnoun,LOC\nx1\tverb,noun\nx2\tverb,noun\nx3\tverb,noun\n"
                        "Newcastle\tGPE,noun,LOC\n")
            out = os.path.join(tmp, "honk.tab")
            honk_table.write_tab(graph, out, HOnK.most_specific_type, ref)
            with open(out) as f:
                rows = self._rows(f.read())
        self.assertEqual(rows["run"], "noun,verb")               # unchanged set: v4.4's string
        self.assertEqual(rows["walk"], "verb,noun")              # not in v4.4's table: that set's commonest order
        self.assertEqual(rows["Haymarket"], "FAC")
        self.assertEqual(rows["Newcastle"], "GPE,FAC,noun")      # a set v4.4 never had: precedence order


class HOnKv2IsKeptApartFromV44(unittest.TestCase):
    """Oliver, 16 and 18 Nov: v4.5's HOnK v2 table goes in a separate database; the shared "honk" table (DB lassi) and
    v4.4's cache/honk_oxstore are never rebuilt. v4.4 named all three in code ("honk", "cache", "LaSSI/HOnK.ttl")."""

    def test_names_are_v45s_own(self):
        from LaSSI.Configuration import HONK_CACHE, HONK_GRAPH, HONK_TABLE
        self.assertEqual((HONK_TABLE, HONK_CACHE, HONK_GRAPH), ("honk_v2", "cache_v45", "LaSSI/HOnK-v2-0-0.nt"))

    def test_the_fuzzy_matcher_reads_the_v2_table(self):
        from LaSSI.external_services.HOnKFuzzyMatch import HOnKFuzzyMatch
        self.assertEqual(HOnKFuzzyMatch(None, None, None).s.tablename, "honk_v2")

    def test_the_database_is_lassi_v45(self):
        import os
        from LaSSI.external_services.utilities.DatabaseConfiguration import load_db_configuration
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(load_db_configuration(os.path.join(root, "connection.yaml")).db, "lassi_v45")


if __name__ == "__main__":
    unittest.main()
