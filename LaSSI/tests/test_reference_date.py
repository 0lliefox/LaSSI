"""The reference date relative time expressions are resolved against (issue 158, 2026-09-25). No pipeline run needed.

The Java wrapper resolves "Monday" or "five days from Monday" against the day the pipeline runs. A dataset can carry
its own date: the LaSSI argument `reference_date`, else `reference_date` in the sidecar `<stem>.meta.yaml` next to
the dataset file (neet/evidence_cases/<case>.meta.yaml, the notice's publication date), else none.
"""
import os
import tempfile
import unittest

from LaSSI.LaSSI import resolve_reference_date
from LaSSI.external_services.AnchoredTime import normalise_reference_date


class NormaliseTests(unittest.TestCase):
    def test_dates_datetimes_and_iso_strings(self):
        import datetime
        self.assertEqual(normalise_reference_date("2026-04-13T00:00:00+01:00"), "2026-04-13")
        self.assertEqual(normalise_reference_date(datetime.date(2026, 4, 13)), "2026-04-13")
        self.assertEqual(normalise_reference_date(datetime.datetime(2026, 4, 13, 23, 59)), "2026-04-13")
        self.assertIsNone(normalise_reference_date(None))
        with self.assertRaises(ValueError):
            normalise_reference_date("13 April")


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.case = os.path.join(self.dir.name, "case.yaml")
        with open(self.case, "w") as fh:
            fh.write("- a sentence\n")

    def tearDown(self):
        self.dir.cleanup()

    def _sidecar(self, text):
        with open(os.path.join(self.dir.name, "case.meta.yaml"), "w") as fh:
            fh.write(text)

    def test_no_sidecar_no_date(self):
        self.assertIsNone(resolve_reference_date(self.case))

    def test_the_sidecar_gives_the_date(self):
        self._sidecar("reference_date: '2026-04-13'\n")
        self.assertEqual(resolve_reference_date(self.case), "2026-04-13")

    def test_the_argument_wins(self):
        self._sidecar("reference_date: '2026-04-13'\n")
        self.assertEqual(resolve_reference_date(self.case, "2026-05-01T09:00:00"), "2026-05-01")


if __name__ == "__main__":
    unittest.main()
