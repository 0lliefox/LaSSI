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


def _v2_graph(cache_dir):
    """A HOnK in memory holding HOnK v2's shape for two statements: "fast oppositeOf1 slow" and "player memberOf1 team",
    each instance a sub-property (and instance) of its relation, oppositeOf a sub-property of neqTo and memberOf of
    partOf, as the release declares them; neqTo and partOf have no statement of their own."""
    import pyoxigraph as ox
    h = HOnK(cache_dir, "u", "p", "localhost", 0, False)
    h.start()
    H, ID = "https://ofox.co.uk/honk#", "https://ofox.co.uk/honk/id/"
    RDF_TYPE = ox.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    LABEL = ox.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    SUB = ox.NamedNode("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")
    n = lambda local: ox.NamedNode(H + local)
    add = lambda s, p, o: h.graph.add(ox.Quad(s, p, o))
    add(n("oppositeOf"), SUB, n("neqTo"))
    add(n("memberOf"), SUB, n("partOf"))
    for inst, rel, s, o in (("oppositeOf1", "oppositeOf", "fast", "slow"), ("memberOf1", "memberOf", "player", "team")):
        add(n(inst), RDF_TYPE, n(rel))
        add(n(inst), SUB, n(rel))
        add(ox.NamedNode(ID + s), n(inst), ox.NamedNode(ID + o))
        for w in (s, o):
            add(ox.NamedNode(ID + w), LABEL, ox.Literal(w))
    return h


class RelationsAreReadWithTheirSubRelations(unittest.TestCase):
    """HOnK v2 has no statement under honk:neqTo itself (antonyms are oppositeOf, 75,285, and distinctFrom, 3,647) and
    puts member meronyms under memberOf (72,089), a sub-property of partOf. v4.4 read only instances typed exactly the
    relation, so on v2 it had no antonyms at all. Oliver, 19 Nov: read both through rdfs:subPropertyOf."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.h = _v2_graph(self.tmp.name)
        self.addCleanup(self.h.stop)

    def test_antonyms_reach_the_adjacency(self):
        self.h._build_relation_adjacency()
        self.assertEqual(self.h._neqTo_adj, {"fast": {"slow"}, "slow": {"fast"}})

    def test_member_meronyms_reach_partof(self):
        self.h._build_relation_adjacency()
        self.assertEqual(self.h._partOf_wholes, {"player": {"team"}})

    def test_tbox_queries_see_sub_relations(self):
        # The TBox rules ask getOutgoingNodes(x, "neqTo") / (x, "partOf"); HOnK answers through the instance query.
        self.assertEqual(self.h._getOutgoingNodesByClassInstance("fast", "neqTo"), {"slow"})
        self.assertEqual(self.h._getOutgoingNodesByClassInstance("player", "partOf"), {"team"})


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
            h = _v2_graph(tmp)
            try:
                h.graph = self._NoLabelScan(h.graph)
                h._build_relation_adjacency()
                self.assertEqual(h._neqTo_adj, {"fast": {"slow"}, "slow": {"fast"}})
                self.assertEqual(h._partOf_wholes, {"player": {"team"}})
            finally:
                h.graph = h.graph._store
                h.stop()


if __name__ == "__main__":
    unittest.main()
