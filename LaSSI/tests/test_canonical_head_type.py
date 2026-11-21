"""The paraphrase-canonical form takes a compound's head as its name, and the head does not inherit the type of the name
it replaces (20 Nov 2026).

City-centre set (test_sentences/nc/newcastle_mdpi.yaml): "It is busy in Newcastle" contradicted "Traffic is flowing in
Newcastle city centre, on Saturdays" and three others. The expansion search compared be[SPACE: Newcastle] with the
canonical form of the other sentence, in which Newcastle[extra: city centre] had become "city centre" typed gpe. The
spatial comparison then read "city centre" as a second settlement beside Newcastle. HOnK is not loaded: its paraphrase
lookups are stubbed and the spatial helpers run as they do when the store is not ready.
"""
import unittest
from unittest import mock

from LaSSI.HOnK.HOnK import HOnKSingleton
import LaSSI.HOnK.TBox.Canonicaliser as C
import LaSSI.HOnK.TBox.SpatialReasoner as SR
from LaSSI.structures.extended_fol.Formulae import FVariable


def _canon(fvar):
    with mock.patch.object(C, "_canon_lookup_paraphrase_concept", lambda name: None), \
            mock.patch.object(C, "_canon_paraphrase_member", lambda name: None):
        return C._canon_fvar(fvar)


class CanonicalHeadType(unittest.TestCase):
    def test_head_takes_the_common_noun_type(self):
        v = _canon(FVariable(name="Newcastle", type="GPE", specification="city centre"))
        self.assertEqual((v.name, v.type, v.specification), ("city centre", "noun", None))

    def test_place_head_of_a_station_name(self):
        v = _canon(FVariable(name="Haymarket", type="LOC", specification="Metro station"))
        self.assertEqual((v.name, v.type), ("metro station", "noun"))

    def test_name_without_a_specification_keeps_its_type(self):
        v = _canon(FVariable(name="Newcastle", type="GPE"))
        self.assertEqual((v.name, v.type), ("newcastle", "gpe"))


class CityCentreIsNotASecondTown(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: False))
        p.start()
        self.addCleanup(p.stop)

    def test_newcastle_and_its_city_centre_are_not_disjoint(self):
        newcastle = FVariable(name="Newcastle", type="GPE")
        centre = _canon(FVariable(name="Newcastle", type="GPE", specification="city centre"))
        self.assertFalse(SR._places_derivably_disjoint({"Newcastle": newcastle}, {centre.name: centre},
                                                       require_unique_on_both=True))

    def test_the_old_type_made_them_two_settlements(self):
        # What the canonical form gave before: the head typed gpe.
        newcastle = FVariable(name="Newcastle", type="GPE")
        centre = FVariable(name="city centre", type="gpe")
        self.assertTrue(SR._places_derivably_disjoint({"Newcastle": newcastle}, {"city centre": centre},
                                                      require_unique_on_both=True))

    def test_two_named_towns_are_still_disjoint(self):
        self.assertTrue(SR._places_derivably_disjoint({"Newcastle": FVariable(name="Newcastle", type="GPE")},
                                                      {"Sunderland": FVariable(name="Sunderland", type="GPE")},
                                                      require_unique_on_both=True))


if __name__ == "__main__":
    unittest.main()
