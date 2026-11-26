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
from LaSSI.structures.extended_fol.Formulae import FAnd, FBinaryPredicate, FNot, FOr, FVariable

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


FORECAST_HONK = SimpleNamespace(
    getGeoSuffixNouns=lambda: set(), getFacilityNouns=lambda: set(), are_synonyms=lambda a, b: False, _isA_supers={},
    # HOnK's PredictionVerb and WeatherConditionNoun/Adjective classes (raw_data/verbs/prediction_verbs.txt,
    # raw_data/nouns/weather_condition_nouns.txt), as far as these statements need them
    getPredictionVerbs=lambda: {"forecast", "predict", "expect", "expected"},
    getWeatherConditionNouns=lambda: {"forecast", "rain", "precipitation", "chance", "wind", "cloud"},
    getWeatherConditionAdjectives=lambda: {"cloudy"})


def forecast_claim(town):
    """"<town> is forecast to be cloudy with a 10% chance of rain at 4pm" as the parser gives it."""
    return FBinaryPredicate("be", FVariable("forecast", "noun"),
                            FAnd((FVariable("rain", "noun", specification="%"), FVariable("cloudy", "JJ"))), 1.0,
                            frozenset({("SPACE", (place(town, "at", "GPE"),))}))


MET_OFFICE = FBinaryPredicate(
    "be", FVariable("Met Office", "GPE", specification="forecast"),
    FAnd((FVariable("precipitation", "noun", specification="chance"), FVariable("wind", "noun"))), 1.0,
    frozenset({("SPACE", (place("Newcastle upon Tyne", "at", "GPE"),))}))


def roadworks(town):
    """"Work expected to end on 17 April in <town>": a prediction term, no weather term."""
    return FBinaryPredicate("end", FVariable("work", "noun"), FVariable("?2", "existential"), 1.0,
                            frozenset({("SPACE", (place(town, "at", "GPE"),)),
                                       ("MODALITY", FVariable("expected", "JJ"))}))


class ForecastsKeepTheirPlace(unittest.TestCase):
    """Policy v3.1's Forecasts rule (Oliver, 24 Nov: "Yes, off for forecasts"; recognised from HOnK's classes, not the
    case or the source): a forecast applies only to its own place and time, so a forecast for another town is NEE.
    CURB's weather_004_R, "Gateshead is forecast to be cloudy ... at 4pm" against the Met Office forecast for Newcastle."""

    def setUp(self):
        for p in (mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: True)),
                  mock.patch.object(HOnKSingleton, "get", staticmethod(lambda: FORECAST_HONK), create=True),
                  mock.patch.object(SR, "_class_typed_members", lambda: {}),
                  mock.patch.object(SR, "_geo_names_compatible", lambda kb, a, b: False)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(SR.set_doc_source, None, None)
        SR.set_doc_source("Met Office", "Newcastle upon Tyne")

    def test_recognised_from_the_classes(self):
        self.assertTrue(SR._is_weather_forecast(forecast_claim("Gateshead")))
        self.assertTrue(SR._is_weather_forecast(MET_OFFICE))
        self.assertFalse(SR._is_weather_forecast(roadworks("Newcastle")))  # a prediction, but not of the weather
        self.assertFalse(SR._is_weather_forecast(recorded(place("Newcastle", "at", "GPE"))))

    def test_forecast_against_a_forecast_for_another_town(self):
        self.assertFalse(SR._space_city_mismatch_contradiction(forecast_claim("Newcastle"), forecast_claim("Gateshead")))

    def test_met_office_forecast_against_another_towns_forecast(self):
        self.assertFalse(SR._space_city_mismatch_contradiction(MET_OFFICE, forecast_claim("Gateshead")))
        self.assertFalse(SR._space_city_mismatch_contradiction(forecast_claim("Gateshead"), MET_OFFICE))
        self.assertFalse(SR._space_mismatch_contradiction(MET_OFFICE, forecast_claim("Gateshead")))

    def test_roadworks_expected_to_end_keep_the_town_test(self):
        self.assertTrue(SR._space_city_mismatch_contradiction(roadworks("Newcastle"), roadworks("Sunderland")))

    def test_a_record_keeps_the_town_test(self):
        self.assertTrue(SR._space_city_mismatch_contradiction(recorded(place("Newcastle", "at", "GPE")),
                                                              recorded(place("Sunderland", "at", "GPE"))))

    def test_a_forecast_for_another_town_is_a_different_place(self):
        # Not a contradiction, but not support either: the score is capped at NEE (DifferentPlaceScoresNEE).
        self.assertTrue(SR._space_places_differ(MET_OFFICE, forecast_claim("Gateshead")))
        self.assertFalse(SR._space_places_differ(MET_OFFICE, forecast_claim("Newcastle")))  # "upon Tyne" variant


def inside(name, rel, ner, town):
    """A place written inside a town, "<name>, <town>": eFOL keeps the town as the place's specification."""
    t = {"on or near": FOr((STAY, NEAR)), "near": "near place", "at": "stay in place"}[rel]
    return FVariable(name, ner, town, None, None, frozenset({("type", t)}))


class PlaceInsideATown(unittest.TestCase):
    """Policy v3.1 (Oliver, 24 Nov): "X, T" is a place inside T, and "GPE first" is for a bare GPE. CURB's crime_003_R:
    Police.uk's "On or near Haymarket, Newcastle", which the parser types Haymarket(GPE)[extra: Newcastle], against
    "near Eldon Square in Newcastle". T counts as a town only when the comparison set types it GPE somewhere."""

    def setUp(self):
        for p in (mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: False)),
                  mock.patch.object(HOnKSingleton, "get", staticmethod(lambda: STUB_HONK), create=True)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(SR.set_doc_source, None, None)
        self.addCleanup(SR.set_doc_gpe_names, ())
        SR.set_doc_source("Police.uk", "On or near Haymarket, Newcastle")
        SR.set_doc_gpe_names({"newcastle", "haymarket"})
        self.ev = recorded(inside("Haymarket", "on or near", "GPE", "Newcastle"))

    def test_the_place_lies_in_its_town(self):
        self.assertEqual(SR._container_of(self.ev.properties and dict(self.ev.properties)["SPACE"][0]), "Newcastle")
        self.assertIsNone(SR._place_kind("Haymarket", inside("Haymarket", "on or near", "GPE", "Newcastle")))
        self.assertEqual(SR._place_kind("Haymarket", place("Haymarket", "near", "GPE")), "settlement")  # bare GPE

    def test_against_another_place_in_the_same_town(self):
        claim = recorded(place("Eldon Square", "near", "FAC"), place("Newcastle", "at", "GPE"))
        self.assertFalse(SR._space_city_mismatch_contradiction(self.ev, claim))
        self.assertFalse(SR._space_mismatch_contradiction(self.ev, claim))
        self.assertTrue(SR._space_places_differ(self.ev, claim))  # NEE, not Supported

    def test_against_another_town(self):
        claim = recorded(place("Sunderland", "at", "GPE"))
        self.assertTrue(SR._space_city_mismatch_contradiction(self.ev, claim))
        self.assertTrue(SR._space_city_mismatch_contradiction(claim, self.ev))

    def test_against_its_own_town(self):
        claim = recorded(place("Newcastle", "at", "GPE"))
        self.assertFalse(SR._space_city_mismatch_contradiction(self.ev, claim))
        self.assertFalse(SR._space_places_differ(self.ev, claim))

    def test_only_a_town_the_comparison_set_types_gpe(self):
        SR.set_doc_gpe_names(())
        self.assertIsNone(SR._container_of(inside("Haymarket", "on or near", "GPE", "Newcastle")))
        SR.set_doc_gpe_names({"newcastle", "met office"})
        self.assertIsNone(SR._container_of(v("Met Office", "GPE", spec="forecast")))  # not typed GPE anywhere
        SR.set_doc_gpe_names({"newcastle", "station"})
        self.assertIsNone(SR._container_of(v("Whitley Bay", "GPE", spec="station")))  # a class noun, even if typed GPE
        self.assertIsNone(SR._container_of(v("Newcastle upon Tyne", "GPE", spec="Newcastle")))  # its own name


def roadworks_on(street, ner):
    """"Roadworks to install a new boundary box on <street>"."""
    return FBinaryPredicate("install", FVariable("?1", "existential"), FVariable("boundary box", "noun"), 1.0,
                            frozenset({("SPACE", (place(street, "at", ner),))}))


class _StubbedCount:
    """HOnK not ready, a roadworks source, and the model count stubbed at full entailment."""

    def setUp(self):
        for p in (mock.patch.object(HOnKSingleton, "isReady", staticmethod(lambda: False)),
                  mock.patch.object(HOnKSingleton, "get", staticmethod(lambda: STUB_HONK), create=True)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(SR.set_doc_source, None, None)
        SR.set_doc_source("Newcastle City Council Roadworks", None)

    @staticmethod
    def score(lhs, rhs):
        from LaSSI.structures.extended_fol.TabularCWASemantics import TabularCWASemantics
        t = TabularCWASemantics.__new__(TabularCWASemantics)
        t.sentence_list = [lhs, rhs]
        with mock.patch.object(TabularCWASemantics, "_raw_id_similarity", lambda self, i, j: 1.0):
            return t.get_straightforward_id_similarity(0, 1)


class DifferentPlaceScoresNEE(_StubbedCount, unittest.TestCase):
    """Policy v3.1 (Oliver, 24 Nov): "a different street gets NEE in the score". CURB's roadworks_007_R: "Roadworks to
    install a new boundary box ... on Meadowdale Crescent" against the same works on Stakeford Lane scored 1.0 under
    b52ab6f, since SPACE is not a BDD variable. The model count is stubbed at full entailment here."""

    def test_another_street_is_nee(self):
        self.assertTrue(SR._space_places_differ(roadworks_on("Stakeford Lane", "LOC"),
                                                roadworks_on("Meadowdale Crescent", "FAC")))
        self.assertEqual(self.score(roadworks_on("Stakeford Lane", "LOC"), roadworks_on("Meadowdale Crescent", "FAC")),
                         0.5)

    def test_the_same_street_is_still_supported(self):
        self.assertEqual(self.score(roadworks_on("Stakeford Lane", "LOC"), roadworks_on("Stakeford Lane", "FAC")), 1.0)

    def test_another_town_is_still_refuted(self):
        self.assertEqual(self.score(recorded(place("Newcastle", "at", "GPE")), recorded(place("Sunderland", "at", "GPE"))),
                         0.0)

    def test_a_police_uk_point_named_by_its_type_is_not_a_different_place(self):
        # suite_c11: "On or near Bus/Coach Station, Leeds" against "near Leeds Bus Station" keeps its score. With HOnK's
        # geo-suffix nouns the near-place rule compares "Bus/Coach" and "Leeds Bus"; the value still names the type.
        SR.set_doc_source("Police.uk", "On Or Near Bus/Coach Station")
        ev = recorded(inside("Bus/Coach Station", "on or near", "LOC", "Leeds"))
        claim = recorded(place("Leeds Bus Station", "near", "LOC"))
        honk = SimpleNamespace(**{**vars(STUB_HONK), "getGeoSuffixNouns": lambda: {"station"}})
        for kb in (STUB_HONK, honk):
            with mock.patch.object(HOnKSingleton, "get", staticmethod(lambda: kb), create=True):
                self.assertFalse(SR._space_places_differ(ev, claim))
                self.assertFalse(SR._space_places_differ(claim, ev))
        SR.set_doc_source("Police.uk", "On Or Near Malvern Road")  # a street is a place, not a type
        self.assertTrue(SR._space_places_differ(recorded(inside("Malvern Road", "on or near", "FAC", "Leeds")),
                                                recorded(place("Balmoral Road", "near", "FAC"))))

    def test_a_place_the_other_side_leaves_out_is_not_a_different_place(self):
        street = recorded(place("Fenkle Street", "at", "FAC"), place("Newcastle", "at", "GPE"))
        town = recorded(place("Newcastle", "at", "GPE"))
        self.assertFalse(SR._space_places_differ(street, town))
        self.assertFalse(SR._space_places_differ(town, street))
        self.assertEqual(self.score(street, town), 1.0)


def works(*parts):
    """Roadworks written as a conjunction, as RewriteKernels writes "Roadworks on X: excavate ... to install ...":
    one be(Roadworks, <verb>)[SPACE: <street>] per (verb, street, NER type), or its negation for a leading "not"."""
    out = []
    for verb, street, ner in parts:
        neg = verb.startswith("not ")
        p = FBinaryPredicate("be", FVariable("roadworks", "noun"), FVariable(verb.removeprefix("not "), "verb"), 1.0,
                             frozenset({("SPACE", (place(street, "at", ner),))}))
        out.append(FNot(p) if neg else p)
    return FAnd(tuple(out))


class ConjunctionPlaces(_StubbedCount, unittest.TestCase):
    """A conjunction's places are those of its parts that are not negated (SpatialReasoner._space_view). CURB's
    roadworks_007_R kept 1.0 after 6b3ea26 (job 2943727): the evidence's target is "excavate ... to install ...", so
    RewriteKernels writes it as FAnd(be(Roadworks, excavate)[SPACE: Stakeford Lane], be(Roadworks, install)[SPACE:
    Stakeford Lane]), and the place tests read a place from a predicate only."""

    def test_a_conjunction_sharing_a_place_is_read_for_it(self):
        ev = works(("excavate", "Stakeford Lane", "LOC"), ("install", "Stakeford Lane", "LOC"))
        claim = roadworks_on("Meadowdale Crescent", "FAC")
        self.assertTrue(SR._space_places_differ(ev, claim))
        self.assertTrue(SR._space_places_differ(claim, ev))
        self.assertEqual(self.score(ev, claim), 0.5)
        self.assertEqual(self.score(claim, ev), 0.5)

    def test_the_same_street_in_a_conjunction_is_still_supported(self):
        ev = works(("excavate", "Stakeford Lane", "LOC"), ("install", "Stakeford Lane", "LOC"))
        self.assertEqual(self.score(ev, roadworks_on("Stakeford Lane", "FAC")), 1.0)

    def test_a_conjunction_whose_parts_name_different_places_has_both(self):
        ev = works(("excavate", "Stakeford Lane", "LOC"), ("install", "Meadowdale Crescent", "LOC"))
        self.assertEqual(self.score(ev, roadworks_on("Meadowdale Crescent", "FAC")), 1.0)  # one of its places
        self.assertEqual(self.score(ev, roadworks_on("Fern Drive", "FAC")), 0.5)

    def test_a_negated_part_does_not_give_the_conjunction_its_place(self):
        ev = works(("excavate", "Stakeford Lane", "LOC"), ("not install", "Meadowdale Crescent", "LOC"))
        self.assertTrue(SR._space_places_differ(ev, roadworks_on("Meadowdale Crescent", "FAC")))

    def test_another_town_in_a_conjunction_is_refuted(self):
        ev = FAnd((recorded(place("Newcastle", "at", "GPE")), recorded(place("Newcastle", "at", "GPE"),
                                                                       place("Fenkle Street", "at", "FAC"))))
        self.assertEqual(self.score(ev, recorded(place("Sunderland", "at", "GPE"))), 0.0)
        self.assertEqual(self.score(ev, recorded(place("Newcastle", "at", "GPE"))), 1.0)

    def test_a_predicate_reads_as_before(self):
        p = roadworks_on("Stakeford Lane", "LOC")
        self.assertIs(SR._space_view(p), p)
        n = FNot(p)
        self.assertIs(SR._space_view(n), n)  # a negated sentence is left as it is
        no_place = FAnd((FBinaryPredicate("be", v("roadworks", "noun"), v("excavate", "verb"), 1.0, frozenset()),))
        self.assertIs(SR._space_view(no_place), no_place)


class ConjunctionThroughTheRewrite(_StubbedCount, unittest.TestCase):
    """From the kernel through RewriteKernels to the score, as LaSSI runs it: "Roadworks on Stakeford Lane: excavate
    ... install ..." against works on another street (CURB roadworks_007) and on the same street. The model count is
    stubbed at full entailment; RewriteKernels gets no HOnK verb lists."""

    @staticmethod
    def rewrite(target, street, ner, marker):
        from collections import defaultdict
        from LaSSI.ner.node_functions import create_props_for_singleton
        from LaSSI.structures.extended_fol.rewrite_kernels import RewriteKernels
        from LaSSI.structures.internal_graph.EntityRelationship import Grouping, Relationship, SetOfSingletons, Singleton

        def node(i, name, t, **props):
            return Singleton(id=i, named_entity=name, properties=create_props_for_singleton(props), min=i, max=i,
                             type=t, confidence=1.0, kernel=None)

        dst = (SetOfSingletons(id=20, type=Grouping.AND, entities=[node(6 + k, x, "verb") for k, x in enumerate(target)],
                               min=6, max=6 + len(target), confidence=1.0)
               if len(target) > 1 else node(6, target[0], "verb"))
        kernel = Singleton(id=100, named_entity="", min=-1, max=-1, type="SENTENCE", confidence=1,
                           properties=create_props_for_singleton({"SPACE": [node(3, street, ner, type="stay in place",
                                                                                 **{"2": marker})]}),
                           kernel=Relationship(source=node(1, "Roadworks", "noun"), target=dst,
                                               edgeLabel=node(2, "be", "verb"), isNegated=False))
        rk = RewriteKernels.__new__(RewriteKernels)
        rk.useId, rk.meu_db_row, rk.obj = False, None, kernel
        rk.e = SimpleNamespace(increaseAndGetExistential=iter(range(1, 100)).__next__)
        rk.p = SimpleNamespace(state_verbs=set(), transitive_verbs=set(), movement_verbs=set(), phrasal_verbs=set(),
                               causative_verbs=set(), semi_modal_verbs=set(), means_verbs=set(),
                               materialisation_verbs=set())
        rk.dmin, rk.dmax, rk.dpos = kernel.update_map(defaultdict(lambda: 10000000), defaultdict(lambda: -1),
                                                      defaultdict(lambda: 10000000))
        return rk.rewrite_kernels(kernel)

    def test_the_rewrite_makes_the_conjunction(self):
        ev = self.rewrite(["excavate", "install"], "Stakeford Lane", "LOC", "on")
        self.assertIsInstance(ev, FAnd)
        self.assertEqual(len(ev.args), 2)

    def test_another_street_scores_nee_through_the_rewrite(self):
        ev = self.rewrite(["excavate", "install"], "Stakeford Lane", "LOC", "on")
        claim = self.rewrite(["underway"], "Meadowdale Crescent", "FAC", "on")
        self.assertEqual(self.score(ev, claim), 0.5)

    def test_the_same_street_scores_as_before_through_the_rewrite(self):
        ev = self.rewrite(["excavate", "install"], "Stakeford Lane", "LOC", "on")
        claim = self.rewrite(["underway"], "Stakeford Lane", "FAC", "on")
        self.assertEqual(self.score(ev, claim), 1.0)


if __name__ == "__main__":
    unittest.main()
