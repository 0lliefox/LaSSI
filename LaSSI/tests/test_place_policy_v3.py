"""Labelling policy v3 for places (Ellis, 23 Nov 2026; Oliver, 24 Nov; a declared change).

A different street name on its own is no longer a contradiction, nor a different station or other facility: within
one town LaSSI measures no distances. A different town still is, for every source, Police.uk included ("in Sunderland"
against "On or near X, Newcastle"). A place the sentence types GPE is read as a town before any HOnK class it is
listed in: Parmenides lists the towns Sunderland, Whitley Bay and South Shields as Metro stations.
On hand-built values and formulas only (no parser, no HOnK load, no CURB or held-out text)."""
import unittest
from types import SimpleNamespace
from unittest import mock

import LaSSI.HOnK.TBox.SpatialReasoner as SR
from LaSSI.HOnK.HOnK import HOnKSingleton
from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate, FOr, FVariable

STAY = FVariable("stay in place", "logical_type", None, None, None, frozenset())
NEAR = FVariable("near place", "logical_type", None, None, None, frozenset())
STUB_HONK = SimpleNamespace(getGeoSuffixNouns=lambda: set(), getFacilityNouns=lambda: set(),
                            are_synonyms=lambda a, b: False, _isA_supers={})


def v(name, ner, spec=None):
    return FVariable(name=name, type=ner, specification=spec)


def place(name, rel, ner):
    t = {"on or near": FOr((STAY, NEAR)), "near": "near place", "at": "stay in place"}[rel]
    return FVariable(name, ner, None, None, None, frozenset({("type", t)}))


def recorded(*places):
    return FBinaryPredicate("record", FVariable("?1", "existential", None, None, None, frozenset()),
                            FVariable("theft", "noun", None, None, None, frozenset()), 1.0,
                            frozenset({("SPACE", tuple(places))}))


def disjoint(lhs, rhs):
    return SR._places_derivably_disjoint({x.name: x for x in lhs}, {x.name: x for x in rhs}, require_unique_on_both=True)


class PlacePolicyV3(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: False)),
                  mock.patch.object(HOnKSingleton, "get", staticmethod(lambda: STUB_HONK), create=True)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(SR.set_doc_source, None, None)
        SR.set_doc_source("Newcastle City Council Roadworks", None)

    def honk_lists(self, members):
        """HOnK ready, with `members` as its place classes (Parmenides' "X isA metro station")."""
        for p in (mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: True)),
                  mock.patch.object(SR, "_class_typed_members", lambda: members),
                  mock.patch.object(SR, "_geo_names_compatible", lambda kb, a, b: False)):
            p.start()
            self.addCleanup(p.stop)

    # a different street, station or facility alone: not a contradiction

    def test_two_streets_are_not_disjoint(self):
        self.assertFalse(disjoint([v("Malvern Road", "FAC")], [v("Balmoral Road", "FAC")]))

    def test_two_named_stations_are_not_disjoint(self):
        self.assertFalse(disjoint([v("Haymarket Metro station", "FAC")], [v("Monument Metro station", "FAC")]))

    def test_two_bare_stations_listed_by_honk_are_not_disjoint(self):
        self.honk_lists({"Haymarket": {"metro station"}, "Monkseaton": {"metro station"}})
        self.assertFalse(disjoint([v("Haymarket", "LOC")], [v("Monkseaton", "LOC")]))

    # a different town: a contradiction

    def test_two_towns_are_disjoint(self):
        self.assertTrue(disjoint([v("Newcastle", "GPE")], [v("Sunderland", "GPE")]))

    def test_a_town_listed_as_a_station_is_still_a_town(self):
        self.honk_lists({"Sunderland": {"metro station"}})
        self.assertEqual(SR._place_kind("Sunderland", v("Sunderland", "GPE")), "settlement")
        self.assertTrue(disjoint([v("Newcastle", "GPE")], [v("Sunderland", "GPE")]))

    def test_a_gpe_named_as_a_station_is_a_station(self):
        # "Whitley Bay Metro station", or Whitley Bay[extra: station], is the station, whatever the parser's type.
        self.assertFalse(disjoint([v("Newcastle", "GPE")], [v("Whitley Bay Metro station", "GPE")]))
        self.assertFalse(disjoint([v("Newcastle", "GPE")], [v("Whitley Bay", "GPE", spec="station")]))

    # Police.uk: the record's street never contradicted, its town now does

    def test_police_uk_record_against_another_town(self):
        SR.set_doc_source("Police.uk", "On or near Edward Place")
        ev = recorded(place("Edward Place", "on or near", "FAC"), place("Newcastle", "at", "GPE"))
        claim = recorded(place("Edward Place", "near", "FAC"), place("Sunderland", "at", "GPE"))
        self.assertTrue(SR._space_city_mismatch_contradiction(ev, claim))
        self.assertTrue(SR._space_city_mismatch_contradiction(claim, ev))

    def test_police_uk_record_against_another_street_in_its_town(self):
        SR.set_doc_source("Police.uk", "On or near Edward Place")
        ev = recorded(place("Edward Place", "on or near", "FAC"), place("Newcastle", "at", "GPE"))
        claim = recorded(place("Fenkle Street", "near", "FAC"), place("Newcastle", "at", "GPE"))
        self.assertFalse(SR._space_city_mismatch_contradiction(ev, claim))
        self.assertFalse(SR._space_mismatch_contradiction(ev, claim))

    def test_a_street_in_its_town_against_the_town(self):
        ev = recorded(place("Fenkle Street", "on or near", "FAC"), place("Newcastle", "at", "GPE"))
        claim = recorded(place("Newcastle", "at", "GPE"))
        self.assertFalse(SR._space_city_mismatch_contradiction(ev, claim))
        self.assertFalse(SR._space_mismatch_contradiction(ev, claim))


if __name__ == "__main__":
    unittest.main()
