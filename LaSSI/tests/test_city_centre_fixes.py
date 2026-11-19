"""The first LaSSI paper's city-centre sentences (test_sentences/nc/newcastle_mdpi.yaml), 19 Nov 2026.

Two of the three fixes are checked here without LaSSI's services (no parser, HOnK store or database):
- a plural weekday is a recurring day, whatever its case (ResolveSingleSentence._recurring_weekday);
- a place or organisation name that runs into a known common-noun compound is cut back at the compound
  (ResolveSingleSentence._trim_overreaching_name), with HOnK's exact-label lookup stubbed.
The third (a negated root verb, SentenceX.assign_kernel) needs the whole pipeline and is checked on the city-centre
set, CURB and the constructed suite."""
import importlib.util
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rss", os.path.join(_HERE, os.pardir, "phases", "ResolveSingleSentence.py"))
rss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rss)


def _tokens(sentence):
    out, i = [], 0
    for w in sentence.split():
        j = sentence.index(w, i)
        out.append((w, j, j + len(w)))
        i = j + len(w)
    return out


class _Ent:
    def __init__(self, sentence, text, type_):
        self.text, self.type = text, type_
        self.start_char = sentence.index(text)
        self.end_char = self.start_char + len(text)


_HONK = {"city centre", "city center", "metro station", "newcastle city council", "bus station"}


def _exact(term):
    return term in _HONK


def _trim(sentence, text, type_="GPE"):
    r = rss._trim_overreaching_name(_Ent(sentence, text, type_), sentence, _tokens(sentence), _exact)
    return None if r is None else (r.text, r.start_char, r.end_char)


class TestRecurringWeekday(unittest.TestCase):
    def test_capitalised_plural_is_recurring(self):
        s = "Traffic is flowing in Newcastle city centre, on Saturdays"
        got = rss._recurring_weekday(s, {"text": "2026-11-14", "type": "DATE", "start_char": 48, "end_char": 57})
        self.assertEqual((got["text"], got["type"]), ("XX-XX-6", "SET"))

    def test_sentence_initial_plural_is_recurring(self):
        got = rss._recurring_weekday("Saturdays have usually busy city centers",
                                     {"text": "2026-11-14", "type": "DATE", "start_char": 0, "end_char": 9})
        self.assertEqual(got["type"], "SET")

    def test_lowercase_set_unchanged(self):
        info = {"text": "XX-XX-6", "type": "SET", "start_char": 28, "end_char": 37}
        self.assertIs(rss._recurring_weekday("In Newcastle city center on saturdays, traffic is flowing", info), info)

    def test_singular_weekday_keeps_its_date(self):
        info = {"text": "2026-11-14", "type": "DATE", "start_char": 3, "end_char": 11}
        self.assertIs(rss._recurring_weekday("on Saturday 14 November", info), info)

    def test_every_weekday(self):
        for day, n in (("Mondays", 1), ("Wednesdays", 3), ("Sundays", 7)):
            got = rss._recurring_weekday(f"on {day}", {"text": "x", "type": "DATE", "start_char": 3,
                                                       "end_char": 3 + len(day)})
            self.assertEqual(got["text"], f"XX-XX-{n}")


class TestTrimOverreachingName(unittest.TestCase):
    def test_newcastle_city_centre(self):
        self.assertEqual(_trim("There is traffic in the Newcastle city centre", "Newcastle city"),
                         ("Newcastle", 24, 33))

    def test_us_spelling(self):
        self.assertEqual(_trim("Newcastle city center does not have traffic", "Newcastle city")[0], "Newcastle")

    def test_name_ending_in_a_capitalised_word_unchanged(self):
        self.assertIsNone(_trim("Footbridges at Whitley Bay Metro station are open", "Whitley Bay Metro", "FAC"))

    def test_lowercase_head_without_a_compound_unchanged(self):
        self.assertIsNone(_trim("Lights at Monkseaton station today", "Monkseaton station", "FAC"))

    def test_capitalised_organisation_unchanged(self):
        self.assertIsNone(_trim("Newcastle City Council roadworks", "Newcastle City Council", "ORG"))

    def test_compound_with_a_function_word_ignored(self):
        self.assertIsNone(_trim("A crime near the Parking Area in Newcastle", "Parking Area"))

    def test_not_a_name_type_unchanged(self):
        self.assertIsNone(_trim("Newcastle city centre", "Newcastle city", "DATE"))

    def test_no_lookup_unchanged(self):
        s = "the Newcastle city centre"
        self.assertIsNone(rss._trim_overreaching_name(_Ent(s, "Newcastle city", "GPE"), s, _tokens(s), None))


if __name__ == "__main__":
    unittest.main()
