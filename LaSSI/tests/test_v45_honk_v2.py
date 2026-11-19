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


if __name__ == "__main__":
    unittest.main()
