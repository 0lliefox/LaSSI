"""The Police.uk reading of "On or near X" (dev-set change after archived v4.4; supervisor's email of 12 Nov 2026).

Police.uk moves each crime to its nearest map point and names that point "On or near X" (data.police.uk, About,
"Location Anonymisation"), so the record gives the vicinity of X. The supervisor's four examples, on hand-built
formulas only (no parser, no HOnK load, no CURB or held-out text):
  1. "On or near Prince Albert Terrace" supports "near Prince Albert Terrace";
  2. it does not support "at Prince Albert Terrace" (the location is approximate);
  3. "On or near Parking Area" does not say which car park, so it is not the same place as a named one;
  4. a different place name does not contradict the record.
Each test also shows the behaviour without the sidecar, so it fails on archived v4.4 (37d3163) where it should.
Example 4 since labelling policy v3 (24 Nov 2026, a declared change; test_place_policy_v3.py): a different street does
not contradict any record, Police.uk or not; a different town contradicts every record, Police.uk included.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import LaSSI.HOnK.TBox.SpatialReasoner as SR
from LaSSI.HOnK.HOnK import HOnKSingleton
from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate, FOr, FVariable

STAY = FVariable("stay in place", "logical_type", None, None, None, frozenset())
NEAR = FVariable("near place", "logical_type", None, None, None, frozenset())
ON_OR_NEAR = FOr((STAY, NEAR))


def place(name, rel, ner="LOC"):
    """A SPACE value: `rel` is "on or near", "near" or "at"."""
    t = {"on or near": ON_OR_NEAR, "near": "near place", "at": "stay in place"}[rel]
    return FVariable(name, ner, None, None, None, frozenset({("type", t)}))


def recorded(*places):
    """"<offence> recorded <places>" as the parser gives it."""
    return FBinaryPredicate("record", FVariable("?1", "existential", None, None, None, frozenset()),
                            FVariable("burglary", "noun", None, None, None, frozenset()), 1.0,
                            frozenset({("SPACE", tuple(places))}))


STUB_HONK = SimpleNamespace(getGeoSuffixNouns=lambda: set(), getFacilityNouns=lambda: set(),
                            are_synonyms=lambda a, b: True, _isA_supers={})


class PoliceUkVicinity(unittest.TestCase):

    def setUp(self):
        patches = [mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: False)),
                   mock.patch.object(HOnKSingleton, "get", staticmethod(lambda: STUB_HONK), create=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(SR.set_doc_source, None, None)

    def police(self, location):
        SR.set_doc_source("Police.uk", location)

    def other_source(self):
        SR.set_doc_source("Leeds City Council Roadworks", None)

    # 1 and 2: the vicinity of X supports "near X", not "at X"

    def test_on_or_near_fully_supports_near(self):
        ev = recorded(place("Prince Albert Terrace", "on or near"))
        claim = recorded(place("Prince Albert Terrace", "near"))
        self.other_source()
        self.assertTrue(SR._space_relation_narrowing(ev, claim))  # elsewhere: partial support only
        self.police("On or near Prince Albert Terrace")
        self.assertFalse(SR._space_relation_narrowing(ev, claim))

    def test_on_or_near_restated_is_full_support(self):
        self.police("On or near Prince Albert Terrace")
        ev = recorded(place("Prince Albert Terrace", "on or near"))
        self.assertFalse(SR._space_relation_narrowing(ev, recorded(place("Prince Albert Terrace", "on or near"))))

    def test_on_or_near_does_not_support_at(self):
        self.police("On or near Prince Albert Terrace")
        ev = recorded(place("Prince Albert Terrace", "on or near"))
        self.assertTrue(SR._space_relation_narrowing(ev, recorded(place("Prince Albert Terrace", "at"))))

    def test_vicinity_reading_needs_the_record_point(self):
        """A place that is not the record's map point keeps the usual reading of "on or near"."""
        self.police("On or near Prince Albert Terrace")
        ev = recorded(place("Grey Street", "on or near"))
        self.assertTrue(SR._space_relation_narrowing(ev, recorded(place("Grey Street", "near"))))

    def test_settlement_after_the_point_is_unaffected(self):
        """"On or near X, Newcastle" against "near X in Newcastle": the city is not narrowed either way."""
        self.police("On or near Prince Albert Terrace")
        ev = recorded(place("Prince Albert Terrace", "on or near"), place("Newcastle", "at", "GPE"))
        claim = recorded(place("Prince Albert Terrace", "near"), place("Newcastle", "at", "GPE"))
        self.assertFalse(SR._space_relation_narrowing(ev, claim))

    # 3: a type of place is not a named place

    def test_type_label_is_not_a_named_place(self):
        ev = [place("Bus/Coach Station", "on or near"), place("Newcastle", "at", "GPE")]
        claim = [place("Newcastle Bus Station", "near")]
        terms = frozenset({"station", "bus station", "bus/coach station"})
        with mock.patch("LaSSI.ner.string_functions._ontology_class_suffix_terms", lambda: terms):
            self.other_source()
            self.assertTrue(SR._spaces_corefer(ev, claim, STUB_HONK))  # elsewhere: the same station
            self.police("On or near Bus/Coach Station")
            self.assertFalse(SR._spaces_corefer(ev, claim, STUB_HONK))
            self.assertFalse(SR._spaces_corefer(claim, ev, STUB_HONK))

    def test_type_label_gets_no_vicinity_reading_for_another_name(self):
        """"On or near Parking Area" against "near the Central Station car park" stays partial even if the names are
        related: the record does not say which car park."""
        self.police("On or near Parking Area")
        ev = recorded(place("Parking Area", "on or near"))
        claim = recorded(place("Central Station car park", "near"))
        with mock.patch.object(SR, "_space_values_name_compatible", lambda a, b: True):
            self.assertTrue(SR._space_relation_narrowing(ev, claim))

    # 4: a different place does not contradict the record

    def test_different_street_does_not_refute(self):
        ev = recorded(place("Grey Street", "on or near"))
        claim = recorded(place("Blackett Street", "near"))
        self.other_source()
        self.assertFalse(SR._space_mismatch_contradiction(ev, claim))  # policy v3: not for any source either
        self.police("On or near Grey Street")
        self.assertFalse(SR._space_mismatch_contradiction(ev, claim))
        self.assertFalse(SR._space_mismatch_contradiction(claim, ev))

    def test_bare_street_name_does_not_refute(self):
        """The archived curated arm read a bare Police.uk name ("Briggate") as a street and refuted another street."""
        self.police("On Or Near Briggate")
        ev = recorded(place("Briggate", "on or near"))
        claim = recorded(place("Boar Lane", "near"))
        with mock.patch.dict("os.environ", {"LASSI_SYNONYMY": "curated"}):
            self.assertFalse(SR._space_mismatch_contradiction(ev, claim))

    def test_different_city_refutes(self):
        """Policy v3 (Oliver, 24 Nov): a claim placing the event in another town than the record's is Refuted, for a
        Police.uk record too. Up to this change the record was never contradicted by any place."""
        ev = recorded(place("Grey Street", "on or near"), place("Newcastle", "at", "GPE"))
        claim = recorded(place("Grey Street", "near"), place("Sunderland", "at", "GPE"))
        self.other_source()
        self.assertTrue(SR._space_city_mismatch_contradiction(ev, claim))  # elsewhere: two settlements
        self.police("On or near Grey Street")
        self.assertTrue(SR._space_city_mismatch_contradiction(ev, claim))


if __name__ == "__main__":
    unittest.main()
