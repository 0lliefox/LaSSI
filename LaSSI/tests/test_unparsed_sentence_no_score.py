"""A sentence LaSSI could not parse has no score (Oliver, 25 Nov 2026: "Fix LaSSI + scorer check").

When building a sentence's form raises, LaSSI._internal_graph logs the error, writes "ERROR" to string_rep and keeps
None as the form. Until now, every None form was the same key in TabularCWASemantics.sentence_to_id, with no atoms, so
the model count read it as true: the evidence "entailed" every unparsed claim at 1.0, i.e. Supported (job 2943728: 23
of 24 weather claims). A pair with an unparsed sentence is now None in the matrix, which the scorers count as no usable
output. These tests run LaSSI's own phases (_internal_graph, _logical_rewriting, _calculate_matrix on the Logical
path) with the parser, the logical rewriting and the model count stubbed; the model count stub answers 1.0 for every
pair, as TabularCWASemantics did for a None form."""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.tests.benchmark import Benchmark


class _Form:
    def __init__(self, text):
        self.text = text

    def to_string(self):
        return self.text


class _Provenance:
    """GraphProvenance for a stubbed parse: the "graph" is the sentence itself, and "UNPARSABLE" raises as
    KernelPostProcessor did on 4f3e7b3's bare TIME entry."""

    def __init__(self, graph, meu_db, simple):
        self.graph = graph

    def internal_graph(self):
        if self.graph == "UNPARSABLE":
            raise TypeError("'Singleton' object is not iterable")
        return _Form(f"form({self.graph})")

    def sentence(self):
        return self.internal_graph()


class _AlwaysEntails:
    """TabularCWASemantics as it read a None form: every pair scores 1.0."""

    def __init__(self, sentence_list, cache_folder):
        self.calls = []

    def __call__(self, x, y):
        self.calls.append((x, y))
        return 1.0


def lassi(folder):
    from LaSSI.LaSSI import LaSSI
    l = LaSSI.__new__(LaSSI)
    l.transformation = SentenceRepresentation.Logical
    l.full_transformation = "Logical"
    l.catabolites_of_dataset = folder
    l.string_rep_dir = os.path.join(folder, "string_rep.txt")
    l.matrix_file = None
    l.useId = False
    l.sentences_benchmark = Benchmark("test_unparsed_sentence_no_score")
    l.logger = lambda *a, **k: None
    return l


def run(sentences):
    """The Logical path from the parse to the matrix; returns (matrix, string_rep text, pairs the model count saw)."""
    with tempfile.TemporaryDirectory() as folder:
        l = lassi(folder)
        l.meu_dbs = [SimpleNamespace(first_sentence=s) for s in sentences]
        seen = {}

        def tabular(sentence_list, cache_folder):
            seen["f"] = _AlwaysEntails(sentence_list, cache_folder)
            return seen["f"]

        # The stubs stand in for whole modules, so their imports (HOnK through Services) never run.
        stubs = {
            "LaSSI.structures.provenance.GraphProvenance": SimpleNamespace(GraphProvenance=_Provenance),
            "LaSSI.structures.extended_fol.rewrite_kernels": SimpleNamespace(rewrite_kernels=lambda x, db, use_id: x.text),
            "LaSSI.structures.extended_fol.TBoxReasoning": SimpleNamespace(TBoxReasoningSingleton=MagicMock()),
            "LaSSI.structures.extended_fol.TabularCWASemantics": SimpleNamespace(TabularCWASemantics=tabular),
        }
        with patch.dict(sys.modules, stubs):
            forms = l._logical_rewriting(l._internal_graph(list(sentences)))
            matrix = l._calculate_matrix(forms)
        with open(l.string_rep_dir, encoding="utf-8") as fh:
            rep = fh.read()
    return matrix, rep, seen["f"].calls


class NoScoreForAnUnparsedSentence(unittest.TestCase):
    def test_an_unparsed_claim_has_no_score(self):
        matrix, rep, calls = run(["evidence", "UNPARSABLE", "claim"])
        self.assertIn("UNPARSABLE ⇒ ERROR", rep)
        self.assertEqual(matrix[0], [1.0, None, 1.0])  # E->C: the unparsed claim is not Supported
        self.assertEqual([row[1] for row in matrix], [None, None, None])
        self.assertEqual(matrix[1], [None, None, None])
        self.assertEqual(matrix[2][0], 1.0)
        self.assertFalse(any(x is None or y is None for x, y in calls))  # the model count never sees a missing form

    def test_an_unparsed_evidence_gives_no_score_at_all(self):
        matrix, _, _ = run(["UNPARSABLE", "claim one", "claim two"])
        self.assertEqual(matrix[0], [None, None, None])
        self.assertEqual([row[0] for row in matrix], [None, None, None])
        self.assertEqual(matrix[1][2], 1.0)

    def test_two_unparsed_claims_are_not_one_sentence(self):
        # Before, both None forms were one key, so the pair between them was the diagonal's 1.0.
        matrix, _, _ = run(["evidence", "UNPARSABLE", "UNPARSABLE"])
        self.assertIsNone(matrix[1][2])
        self.assertIsNone(matrix[2][1])

    def test_parsed_sentences_score_as_before(self):
        matrix, rep, calls = run(["evidence", "claim"])
        self.assertNotIn("ERROR", rep)
        self.assertEqual(matrix, [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
