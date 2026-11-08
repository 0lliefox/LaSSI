"""Regression tests for the v4.3 -> v4.4 corrections (CHANGES_v4.3_to_v4.4.md). No pipeline run, no HOnK load.

Each test states the general rule a change implements, on constructed input only: the parser output of eight
constructed sentences (fixtures/v44_constructed_modal.yaml) and hand-built formulas. No CURB or held-out text.
"""
import os
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from LaSSI.HOnK.HOnK import CasusHappening, HOnK
from LaSSI.HOnK.TBox.ComparatorUtils import transformCaseWhenOneArgIsNegated
from LaSSI.HOnK.TBox.ConstituentComparator import _formula_core_values
from LaSSI.phases.DependencyNormaliser import (_number_by_position, _takes_infinitive_complement, normalise_gsm,
                                               parse_gsm, serialise_gsm)
from LaSSI.structures.DependencyRoles import is_infinitive_marker
from LaSSI.structures.extended_fol.Enums import PairwiseCases
from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate, FNot, FVariable
from LaSSI.structures.extended_fol.TabularCWASemantics import TabularCWASemantics
from LaSSI.structures.kernels.transitivity import drops_object

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURE = os.path.join(HERE, "fixtures", "v44_constructed_modal.gsmDB.txt")
MODAL_ADJECTIVES = {"unlikely", "likely", "possible", "expected"}

# Sentence order in the fixture (fixtures/v44_constructed_modal.yaml).
BARE, PASSIVE, WITH_BARE, WITH_CONSIDERED, CONJ_CONSIDERED, CONJ_BARE, PURPOSE, COMPLEMENT = range(8)


def _blocks(text):
    return [nodes for _, nodes, _ in parse_gsm(text)]


def _by_word(nodes):
    return {n.xi[0]: n for n in nodes if n.xi}


def _edges(node, nodes):
    ids = {n.id: n.xi[0] for n in nodes if n.xi}
    return {(label, ids[t]) for label, targets in node.phi for _, t in targets}


class DependencyNormaliserTests(unittest.TestCase):
    """cab8fbc: structural rewrites before the grammar; the parser output itself is untouched."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as fh:
            cls.raw = fh.read()
        cls.before = _blocks(cls.raw)
        cls.numbered = parse_gsm(cls.raw)
        for _, nodes, _ in cls.numbered:
            if nodes:
                _number_by_position(nodes)
        cls.after = _blocks(normalise_gsm(cls.raw, MODAL_ADJECTIVES))

    def test_round_trip_is_byte_identical(self):
        self.assertEqual(serialise_gsm(parse_gsm(self.raw)), self.raw)

    def test_no_modal_adjectives_only_numbers_in_token_order(self):
        self.assertEqual(normalise_gsm(self.raw, set()), serialise_gsm(self.numbered))

    def test_judgement_passive_takes_the_shape_of_the_bare_form(self):
        # "Delays are considered unlikely." -> "Delays are unlikely." (the issuing authority is the judge)
        after = _by_word(self.after[PASSIVE])
        self.assertNotIn("considered", after)
        self.assertEqual(_edges(after["unlikely"], self.after[PASSIVE]),
                         _edges(_by_word(self.before[BARE])["unlikely"], self.before[BARE]))

    def test_bare_form_is_unchanged(self):
        self.assertEqual(serialise_gsm([("", self.after[BARE], "")]), serialise_gsm([("", self.numbered[BARE][1], "")]))

    def test_coordinated_judgement_passive_is_collapsed(self):
        # "Roadworks are underway, and delays are considered unlikely."
        after = _by_word(self.after[CONJ_CONSIDERED])
        self.assertNotIn("considered", after)
        self.assertIn(("nsubj", "delays"), _edges(after["unlikely"], self.after[CONJ_CONSIDERED]))
        self.assertIn(("conj", "unlikely"), _edges(after["underway"], self.after[CONJ_CONSIDERED]))

    def test_judgement_relative_becomes_a_modifier_of_the_noun(self):
        # "..., with delays considered unlikely." -> "with unlikely delays"
        after = _by_word(self.after[WITH_CONSIDERED])
        self.assertNotIn("considered", after)
        self.assertIn(("amod", "unlikely"), _edges(after["delays"], self.after[WITH_CONSIDERED]))

    def test_node_ids_stay_contiguous(self):
        for nodes in self.after:
            self.assertEqual(sorted(int(n.id) for n in nodes), list(range(len(nodes))))

    def test_passive_purpose_is_an_adverbial_clause(self):
        # "The road is closed to allow resurfacing.": "close" has no clausal-complement frame in WordNet.
        closed = _by_word(self.after[PURPOSE])["closed"]
        self.assertIn(("advcl", "allow"), _edges(closed, self.after[PURPOSE]))
        self.assertNotIn(("xcomp", "allow"), _edges(closed, self.after[PURPOSE]))

    def test_raising_complement_stays_a_complement(self):
        # "Trains are expected to run normally.": "expect" takes an infinitive complement.
        expected = _by_word(self.after[COMPLEMENT])["expected"]
        self.assertIn(("xcomp", "run"), _edges(expected, self.after[COMPLEMENT]))

    def test_clausal_complement_frames_from_wordnet(self):
        self.assertIs(_takes_infinitive_complement("close"), False)
        for verb in ("expect", "forecast", "believe", "plan"):
            self.assertIs(_takes_infinitive_complement(verb), True, verb)


class _Node:
    def __init__(self, type):
        self.type = type


class InfinitiveMarkerTests(unittest.TestCase):
    """11c5dfa: the infinitive "to" of a verb is not the preposition "to"."""

    def test_mark_to_on_a_verb_is_the_infinitive_marker(self):
        self.assertTrue(is_infinitive_marker(_Node("verb"), "mark", "to"))

    def test_preposition_to_is_not(self):
        self.assertFalse(is_infinitive_marker(_Node("verb"), "case", "to"))
        self.assertFalse(is_infinitive_marker(_Node("noun"), "mark", "to"))
        self.assertFalse(is_infinitive_marker(_Node("verb"), "mark", "that"))


class NegationTests(unittest.TestCase):
    """cc40ab5: "not A" against B, given the relation between A and B."""

    def test_not_a_excludes_an_equivalent_b(self):
        self.assertEqual(transformCaseWhenOneArgIsNegated(CasusHappening.EQUIVALENT), CasusHappening.EXCLUSIVES)

    def test_not_a_says_nothing_about_a_contrary_b(self):
        # "not at Haymarket" does not put the event at Monument.
        self.assertEqual(transformCaseWhenOneArgIsNegated(CasusHappening.EXCLUSIVES), CasusHappening.INDIFFERENT)

    def test_implications_do_not_carry_over(self):
        for v in (CasusHappening.GENERAL_IMPLICATION, CasusHappening.INSTANTIATION_IMPLICATION,
                  CasusHappening.LOSE_SPEC_IMPLICATION, CasusHappening.MISSING_1ST_IMPLICATION):
            self.assertEqual(transformCaseWhenOneArgIsNegated(v), CasusHappening.INDIFFERENT, v)


class ExclusionModelCountTests(unittest.TestCase):
    """An exclusion is "not both" unless both atoms are un-negated states of a two-state status dimension. A negated
    atom is ~a in the BDD, so XOR(~a, b) would assert a <=> b: "no rain" against "a shower" would make all rain a
    shower."""

    @staticmethod
    def _semantics(x, y, x_node, y_node):
        from dd.autoref import BDD
        t = TabularCWASemantics.__new__(TabularCWASemantics)
        t.minimal_constituents = SimpleNamespace(fromId={0: x, 1: y}.get)
        t._bdd = BDD()
        t._bdd.declare("a", "b")
        t._bdd_for_id_cache = {0: x_node(t._bdd), 1: y_node(t._bdd)}
        t.determine = lambda i, j: PairwiseCases.ConflictingImplication
        return t

    @staticmethod
    def _models(t, constraints):
        return {(m["a"], m["b"]) for m in t._bdd.pick_iter(constraints, care_vars=["a", "b"])}

    def test_not_a_against_a_more_specific_b_is_not_both(self):
        rain, shower = (FVariable(w, "noun", None, None, None, frozenset()) for w in ("rain", "shower"))
        t = self._semantics(FNot(rain), shower, lambda d: ~d.var("a"), lambda d: d.var("b"))
        self.assertEqual(self._models(t, t._universal_truth_bdd({0}, {1})),
                         {(False, False), (True, False), (True, True)})  # rain without a shower is a model

    def test_two_complementary_states_are_exactly_one(self):
        found, unfound = (FVariable(w, "verb", None, None, None, frozenset()) for w in ("found", "unfound"))
        t = self._semantics(found, unfound, lambda d: d.var("a"), lambda d: d.var("b"))
        with mock.patch("LaSSI.HOnK.TBox.LifecycleManager._lifecycle_complementary", return_value=True):
            self.assertEqual(self._models(t, t._universal_truth_bdd({0}, {1})), {(False, True), (True, False)})

    def test_negated_atoms_are_never_complementary(self):
        found, unfound = (FVariable(w, "verb", None, None, None, frozenset()) for w in ("found", "unfound"))
        t = self._semantics(FNot(found), unfound, lambda d: ~d.var("a"), lambda d: d.var("b"))
        with mock.patch("LaSSI.HOnK.TBox.LifecycleManager._lifecycle_complementary", return_value=True):
            self.assertFalse(t._is_complementary_pair(0, 1))

    def test_explanation_tables_list_the_counted_rows(self):
        rain, shower = (FVariable(w, "noun", None, None, None, frozenset()) for w in ("rain", "shower"))
        t = self._semantics(FNot(rain), shower, lambda d: ~d.var("a"), lambda d: d.var("b"))
        rows = lambda df: set(map(tuple, df[["0", "1"]].values.tolist()))
        self.assertEqual(rows(t._mutual_truth(0, 1)), {(0, 0), (0, 1), (1, 0)})
        t = self._semantics(FNot(rain), rain, lambda d: ~d.var("a"), lambda d: d.var("a"))
        self.assertEqual(rows(t._mutual_truth(0, 1)), {(0, 1), (1, 0)})  # a literal pair: exactly one


def _stub_honk(eq=(), neq=(), form_of=(), mode="capped"):
    """A HOnK with only the adjacency tables name_eq reads, built from undirected pairs."""
    kb = HOnK.__new__(HOnK)

    def adj(pairs):
        out = {}
        for a, b in pairs:
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
        return out

    kb._eq_adj, kb._neqTo_adj, kb._formOf_adj = adj(eq), adj(neq), adj(form_of)
    kb._isA_supers = {}
    kb._synonymy_mode = mode
    kb.getSynonymy = lambda k: {k} | kb._eq_adj.get(k, set())
    kb._cached_location_variants = lambda k: ({k}, False)
    kb.getSuperTypes = lambda k: set()
    return kb


class HOnKRelationTests(unittest.TestCase):
    """40db029 and formOf: name_eq reads HOnK's relations as HOnK defines them (thesis Ch. 4 glossary)."""

    def test_eq_and_neqto_on_the_same_pair_gives_no_verdict(self):
        kb = _stub_honk(eq=[("happen", "occur")], neq=[("happen", "occur")])
        self.assertEqual(kb.name_eq("happen", "occur"), CasusHappening.INDIFFERENT)

    def test_neqto_alone_is_exclusive(self):
        kb = _stub_honk(neq=[("open", "closed")])
        self.assertEqual(kb.name_eq("open", "closed"), CasusHappening.EXCLUSIVES)

    def test_a_variant_form_is_the_same_word(self):
        # formOf = "variant form of the same word": an inflection is equivalent even with no eq edge.
        kb = _stub_honk(form_of=[("offences", "offence")])
        self.assertEqual(kb.name_eq("offences", "offence"), CasusHappening.EQUIVALENT)
        self.assertEqual(kb.name_eq("offence", "offences"), CasusHappening.EQUIVALENT)

    def test_curated_equivalence_needs_a_stored_link(self):
        kb = _stub_honk(eq=[("newcastle", "toon"), ("newcastle upon tyne", "toon")], mode="curated")
        self.assertEqual(kb.name_eq("newcastle", "toon"), CasusHappening.EQUIVALENT)
        self.assertEqual(kb.name_eq("newcastle", "newcastle upon tyne"), CasusHappening.INDIFFERENT)

    def test_curated_antonyms_do_not_chain(self):
        kb = _stub_honk(eq=[("get", "begin")], neq=[("begin", "end")], mode="curated")
        self.assertEqual(kb.name_eq("begin", "end"), CasusHappening.EXCLUSIVES)
        self.assertEqual(kb.name_eq("get", "end"), CasusHappening.INDIFFERENT)

    def test_capped_arm_keeps_overlap(self):
        kb = _stub_honk(eq=[("a", "b"), ("c", "b")])
        self.assertEqual(kb.name_eq("a", "c"), CasusHappening.EQUIVALENT)

    def test_a_named_argument_entails_an_existential_not_the_reverse(self):
        kb = _stub_honk()
        self.assertEqual(kb.name_eq("burst pipe", "?3"), CasusHappening.EQUIVALENT)
        self.assertEqual(kb.name_eq("?3", "burst pipe"), CasusHappening.MISSING_1ST_IMPLICATION)
        self.assertEqual(kb.name_eq("?3", "?4"), CasusHappening.EQUIVALENT)

    def test_curated_tbox_rules_read_the_curated_tables_one_hop(self):
        kb = _stub_honk(eq=[("rain", "precipitation"), ("precipitation", "downfall")], neq=[("fast", "slow")],
                        mode="curated")
        self.assertEqual(kb.getOutgoingNodes("rain", "eq"), {"precipitation"})
        self.assertEqual(kb.getOutgoingNodes("Fast", "neqTo"), {"slow"})


class ExistentialCoreTests(unittest.TestCase):
    """Pending fix: an existential argument names nothing, so it cannot cover a property value of the other side."""

    def test_existential_arguments_are_not_cores(self):
        agent = FVariable("?1", "existential", None, None, None, frozenset())
        theft = FVariable("theft", "noun", None, None, None, frozenset())
        # "A theft was recorded" = record(?1, theft)
        self.assertEqual(_formula_core_values(FBinaryPredicate("record", agent, theft, 1.0, frozenset())), [theft])


def _active_rules(path):
    with open(path, encoding="utf-8") as fh:
        text = re.sub(r"/\*.*?\*/", "", fh.read(), flags=re.S)
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))


class TBoxRuleTests(unittest.TestCase):
    """1e48161: no TBox rule derives an implication or equivalence from relatedTo ("general semantic association")."""

    def test_no_active_rule_uses_related_to(self):
        for name in ("query_impl.txt", "query_eq.txt"):
            self.assertNotIn("relatedTo", _active_rules(os.path.join(REPO, name)), name)

    def test_rule_counts(self):
        count = lambda name: len(re.findall(r"(?m)^\s*(?:nested\s+)?match\b", _active_rules(os.path.join(REPO, name))))
        self.assertEqual(count("query_impl.txt"), 25)
        self.assertEqual(count("query_eq.txt"), 19)


class TransitivityTests(unittest.TestCase):
    """The copula keeps its predicate; only an edge with no lexical verb loses its target."""

    def test_the_copula_keeps_its_target(self):
        self.assertFalse(drops_object({"be"}, {"be", "have"}))

    def test_no_lexical_verb_drops_the_target(self):
        self.assertTrue(drops_object({""}, {"be", "have"}))
        self.assertTrue(drops_object(set(), {"be", "have"}))


if __name__ == "__main__":
    unittest.main()
