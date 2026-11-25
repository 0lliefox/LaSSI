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
from LaSSI.structures.extended_fol.Formulae import FAnd, FBinaryPredicate, FOr, FVariable

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


class DifferentPlaceScoresNEE(unittest.TestCase):
    """Policy v3.1 (Oliver, 24 Nov): "a different street gets NEE in the score". CURB's roadworks_007_R: "Roadworks to
    install a new boundary box ... on Meadowdale Crescent" against the same works on Stakeford Lane scored 1.0 under
    b52ab6f, since SPACE is not a BDD variable. The model count is stubbed at full entailment here."""

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


if __name__ == "__main__":
    unittest.main()
