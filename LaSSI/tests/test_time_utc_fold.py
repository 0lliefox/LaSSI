"""Policy v3.1: a claim's time is read in the zone it states. "at 5pm UTC on 14 April 2026" parses as two TIME
entries, the date's ISO value (SUTime folds the clock time into it) and a raw "5pm UTC"; TimeCanonicalisationRule
folds the raw one into the ISO entry that holds its UTC clock time (CURB's corrected weather claims, job 2943465)."""
import unittest

from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.ner.structural_rewrites.time_canonicalisation import TimeCanonicalisationRule, _utc_clock
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


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
        self.assertIsInstance(out, Singleton)  # as "at 5pm on 14 April 2026" parses: one TIME value, not a list
        self.assertEqual(out.named_entity, "2026-04-14T17:00:00Z")
        self.assertEqual(dict(out.properties), {"18": "on", "14": "at", "type": "defined"})

    def test_either_order(self):
        out = rewrite([time("2pm UTC", **{"12": "at", "type": "defined"}),
                       time("2026-04-13T14:00:00Z", **{"16": "on", "type": "defined"})])
        self.assertEqual(out.named_entity, "2026-04-13T14:00:00Z")
        self.assertEqual(dict(out.properties)["12"], "at")

    def test_noon(self):
        out = rewrite([time("2026-04-13T12:00:00Z", **{"12": "on"}), time("noon UTC", **{"9": "at"})])
        self.assertEqual(out.named_entity, "2026-04-13T12:00:00Z")

    def test_a_non_canonical_iso_entry_is_canonicalised_first(self):
        out = rewrite([time("2026-04-13T11-0", **{"13": "on"}), time("11am UTC", **{"9": "at", "nummod": "11"})])
        self.assertEqual(out.named_entity, "2026-04-13T11:00:00Z")
        self.assertNotIn("nummod", dict(out.properties))

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


if __name__ == "__main__":
    unittest.main()
