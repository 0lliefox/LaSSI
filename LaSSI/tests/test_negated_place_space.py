"""A place against "not" that place (20 Nov 2026, city-centre set).

"There is traffic in the Newcastle city centre" is be(traffic)[SPACE: city centre, SPACE: Newcastle] and "Newcastle
has traffic but not in the city centre" is be(traffic)[SPACE: NOT(city centre), SPACE: Newcastle]. The expansion
search matched the shared have(Newcastle, traffic) and dropped the exclusion, so the pair came out Equivalent; the
paper's matrix has a contradiction. SpatialReasoner._space_negated_place_contradiction finds it on the originals.
No LaSSI services are used: the comparator's two tests are stubbed."""
import unittest

from LaSSI.HOnK.HOnK import CasusHappening
from LaSSI.HOnK.TBox.SpatialReasoner import _space_negated_place_contradiction as contradiction
from LaSSI.structures.extended_fol.Formulae import FNot, FUnaryPredicate, FVariable


def var(name, type_="LOC"):
    return FVariable(name=name, type=type_, specification=None, cop=None, id=None)


def be(*space, arg="traffic"):
    return FUnaryPredicate(rel="be", arg=var(arg, "noun"), score=1.0,
                           properties=frozenset(("SPACE", v) for v in space))


CENTRE, NEWCASTLE, MONUMENT = var("city centre"), var("Newcastle", "GPE"), var("Monument")


def same_place(p, n):
    """compare_variable(p, NOT(q)) as the comparator gives it: exclusive when p and q are the same place."""
    return CasusHappening.EXCLUSIVES if p.name == n.arg.name else CasusHappening.INDIFFERENT


def exclusive(x, y):
    return CasusHappening.EXCLUSIVES


def equivalent(x, y):
    return CasusHappening.EQUIVALENT


class TestNegatedPlace(unittest.TestCase):
    def test_city_centre_against_not_city_centre(self):
        x, y = be(CENTRE, NEWCASTLE), be(FNot(CENTRE), NEWCASTLE)
        self.assertTrue(contradiction(x, y, same_place, exclusive))
        self.assertTrue(contradiction(y, x, same_place, exclusive))

    def test_a_different_place_is_not_a_contradiction(self):
        self.assertFalse(contradiction(be(MONUMENT), be(FNot(CENTRE)), same_place, exclusive))

    def test_no_negated_place(self):
        self.assertFalse(contradiction(be(CENTRE), be(NEWCASTLE), same_place, exclusive))

    def test_the_originals_must_describe_one_event(self):
        # The direct comparison decides whether relation and arguments match; when it does not exclude, nothing is said.
        self.assertFalse(contradiction(be(CENTRE), be(FNot(CENTRE)), same_place, equivalent))

    def test_space_in_one_tuple(self):
        x = FUnaryPredicate(rel="be", arg=var("traffic", "noun"), score=1.0,
                            properties=frozenset({("SPACE", (CENTRE, NEWCASTLE))}))
        y = FUnaryPredicate(rel="be", arg=var("traffic", "noun"), score=1.0,
                            properties=frozenset({("SPACE", (FNot(CENTRE), NEWCASTLE))}))
        self.assertTrue(contradiction(x, y, same_place, exclusive))

    def test_negated_predicate_not_unwrapped(self):
        # NOT(be(traffic)[SPACE: city centre]) has its place inside the negation; it is not a positive place.
        self.assertFalse(contradiction(FNot(be(CENTRE)), be(FNot(CENTRE)), same_place, exclusive))


if __name__ == "__main__":
    unittest.main()
