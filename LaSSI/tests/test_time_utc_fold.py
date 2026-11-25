"""Policy v3.1: a claim's time is read in the zone it states. "at 5pm UTC on 14 April 2026" parses as two TIME
entries, the date's ISO value (SUTime folds the clock time into it) and a raw "5pm UTC"; TimeCanonicalisationRule
folds the raw one into the ISO entry that holds its UTC clock time (CURB's corrected weather claims, job 2943465).
TIME stays a list: 4f3e7b3 made a one-entry result a bare entry, and KernelPostProcessor's post-rewrite dedupe, which
iterates every kernel property, failed on 23 of 24 weather claims (job 2943728). PipelineThroughDedupe runs that step."""
import unittest
from unittest.mock import patch

from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.ner.structural_rewrites import RewriteContext, default_registry
from LaSSI.ner.structural_rewrites.time_canonicalisation import TimeCanonicalisationRule, _utc_clock
from LaSSI.structures.internal_graph.EntityRelationship import Relationship, Singleton


def time(name, **props):
    return Singleton(id=abs(hash(name)) % 10000, named_entity=name, properties=create_props_for_singleton(props),
                     min=0, max=0, type="TIME", confidence=1.0)


def kernel(times):
    return Singleton(id=1, named_entity="be", properties=create_props_for_singleton({"TIME": list(times)}),
                     min=0, max=0, type="verb", confidence=1.0)


def rewrite(times):
    rule, k = TimeCanonicalisationRule(), kernel(times)
    bindings = rule.matches(k, None)
    return dict((rule.apply(k, bindings, None) if bindings is not None else k).properties)["TIME"]


class UtcClockReading(unittest.TestCase):
    def test_clock_times_stated_in_utc(self):
        self.assertEqual(_utc_clock("5pm UTC"), (17, 0))
        self.assertEqual(_utc_clock("11am UTC"), (11, 0))
        self.assertEqual(_utc_clock("11:30am GMT"), (11, 30))
        self.assertEqual(_utc_clock("noon UTC"), (12, 0))
        self.assertEqual(_utc_clock("midnight UTC"), (0, 0))
        self.assertEqual(_utc_clock("12am UTC"), (0, 0))
        self.assertEqual(_utc_clock("12pm UTC"), (12, 0))

    def test_not_a_clock_time_in_utc(self):
        self.assertIsNone(_utc_clock("5pm"))
        self.assertIsNone(_utc_clock("5pm BST"))
        self.assertIsNone(_utc_clock("13pm UTC"))
        self.assertIsNone(_utc_clock("2026-04-14T17:00:00Z"))
        self.assertIsNone(_utc_clock(None))


class UtcClockFold(unittest.TestCase):
    def test_the_clock_time_folds_into_the_iso_entry(self):
        out = rewrite([time("2026-04-14T17:00:00Z", **{"18": "on", "type": "defined"}),
                       time("5pm UTC", **{"14": "at", "type": "defined"})])
        self.assertIsInstance(out, tuple)  # a list of one, as the dedup leaves "at 5pm on 14 April 2026"
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].named_entity, "2026-04-14T17:00:00Z")
        self.assertEqual(dict(out[0].properties), {"18": "on", "14": "at", "type": "defined"})

    def test_either_order(self):
        out = rewrite([time("2pm UTC", **{"12": "at", "type": "defined"}),
                       time("2026-04-13T14:00:00Z", **{"16": "on", "type": "defined"})])
        self.assertEqual([t.named_entity for t in out], ["2026-04-13T14:00:00Z"])
        self.assertEqual(dict(out[0].properties)["12"], "at")

    def test_noon(self):
        out = rewrite([time("2026-04-13T12:00:00Z", **{"12": "on"}), time("noon UTC", **{"9": "at"})])
        self.assertEqual([t.named_entity for t in out], ["2026-04-13T12:00:00Z"])

    def test_a_non_canonical_iso_entry_is_canonicalised_first(self):
        out = rewrite([time("2026-04-13T11-0", **{"13": "on"}), time("11am UTC", **{"9": "at", "nummod": "11"})])
        self.assertEqual([t.named_entity for t in out], ["2026-04-13T11:00:00Z"])
        self.assertNotIn("nummod", dict(out[0].properties))

    def test_a_clock_time_no_iso_entry_holds_is_kept(self):
        out = rewrite([time("2026-04-14T22:00:00Z", **{"18": "on"}), time("5pm UTC", **{"14": "at"})])
        self.assertEqual(sorted(t.named_entity for t in out), ["2026-04-14T22:00:00Z", "5pm UTC"])

    def test_another_zone_is_not_folded(self):
        out = rewrite([time("2026-04-14T17:00:00Z", **{"18": "on"}), time("5pm BST", **{"14": "at"})])
        self.assertEqual(sorted(t.named_entity for t in out), ["2026-04-14T17:00:00Z", "5pm BST"])

    def test_a_list_without_utc_keeps_its_shape(self):
        out = rewrite([time("2026-04-14T17:00:00Z", **{"14": "at"}), time("2026-04-14T17-0", **{"nummod": "2026"})])
        self.assertIsInstance(out, tuple)
        self.assertEqual([t.named_entity for t in out], ["2026-04-14T17:00:00Z"])


def node(i, name, type_, **props):
    return Singleton(id=i, named_entity=name, properties=create_props_for_singleton(props), min=i, max=i, type=type_,
                     confidence=1.0, kernel=None)


def claim(times):
    """CURB weather_001's second claim as it reaches the post-rewrite phase: "Newcastle is forecast to have a sunny
    day with a very low chance of rain at 11am UTC on 13 April 2026" -> be(forecast, day)[SPACE, TIME]."""
    space = node(1, "Newcastle", "GPE", type="stay in place")
    return Singleton(id=100, named_entity="", properties=create_props_for_singleton({"SPACE": [space], "TIME": times}),
                     min=-1, max=-1, type="SENTENCE", confidence=1,
                     kernel=Relationship(source=node(2, "forecast", "noun"), target=node(5, "day", "noun", amod="sunny"),
                                         edgeLabel=node(3, "be", "verb"), isNegated=False))


class _Empty:
    """Every lookup is empty: no HOnK class terms, no location-like nodes, no dependency children."""
    def __getattr__(self, name):
        return lambda *a, **k: frozenset()


class _NoHOnKServices:
    def getHOnK(self):
        return _Empty()


class PipelineThroughDedupe(unittest.TestCase):
    """The post_logical_rewrite rules (the registry LaSSI uses), then KernelPostProcessor.remove_duplicate_properties,
    in the order of KernelPostProcessor.run. No LaSSI services: HOnK's class lists and the matchers are empty."""

    @staticmethod
    def post_process(kernel):
        # KernelPostProcessor's import reads HOnK through Services (HOnKLogicalRewriting), as in test_structural_rewrites.
        with patch("LaSSI.external_services.Services.Services.getInstance", return_value=_NoHOnKServices()):
            from LaSSI.ner.KernelPostProcessor import KernelPostProcessor
        kpp = KernelPostProcessor.__new__(KernelPostProcessor)
        kpp.structural_rewrites = default_registry()
        kpp._rewrite_ctx = RewriteContext(node_functions=None, services=_NoHOnKServices(), matchers=_Empty())
        return kpp.remove_duplicate_properties(kpp._apply_structural_rules(kernel, "post_logical_rewrite"))

    def test_a_claim_in_utc_survives_the_dedupe(self):
        out = self.post_process(claim([time("2026-04-13T11:00:00Z", **{"20": "on", "type": "defined"}),
                                       time("11am UTC", **{"16": "at", "type": "defined"})]))
        t = dict(out.properties)["TIME"]
        self.assertEqual([x.named_entity for x in t], ["2026-04-13T11:00:00Z"])
        self.assertEqual(dict(t[0].properties), {"20": "on", "16": "at", "type": "defined"})

    def test_the_same_claim_without_utc_has_the_same_time(self):
        utc = self.post_process(claim([time("2026-04-13T11:00:00Z", **{"20": "on", "type": "defined"}),
                                       time("11am UTC", **{"16": "at", "type": "defined"})]))
        plain = self.post_process(claim([time("2026-04-13T11-0", **{"20": "on", "16": "at", "type": "defined"}),
                                         time("2026-04-13T11-0", **{"nummod": "2026", "type": "defined"})]))
        self.assertEqual([(x.named_entity, dict(x.properties)) for x in dict(utc.properties)["TIME"]],
                         [(x.named_entity, dict(x.properties)) for x in dict(plain.properties)["TIME"]])


if __name__ == "__main__":
    unittest.main()
