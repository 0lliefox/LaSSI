"""existential_locative_to_have on a plain existential (19 Nov 2026, city-centre set).

"There is traffic in the Newcastle city centre" reaches the rule as be(?existential, traffic)[SPACE: Newcastle[..]];
it now becomes have(Newcastle[..], traffic), the form "Newcastle city centre has traffic" already has. A negated or
coordinated SPACE, a place as target, and a non-copula verb are left alone, and the original SPECIFICATION shape
(transport_002) is unchanged. No LaSSI services are used: the rule's location test falls back to the node type."""
import unittest

from LaSSI.ner.structural_rewrites.base import RewriteContext
from LaSSI.ner.structural_rewrites.existential_locative_to_have import ExistentialLocativeToHaveRule
from LaSSI.structures.internal_graph.EntityRelationship import Grouping, Relationship, SetOfSingletons, Singleton


def node(i, name, type_, props=()):
    return Singleton(id=i, named_entity=name, properties=frozenset(props), min=i, max=i, type=type_,
                     confidence=1.0, kernel=None)


def kernel(edge, source, target, props, negated=False):
    return Singleton(id=100, named_entity="", properties=frozenset(props), min=-1, max=-1, type="SENTENCE",
                     confidence=1, kernel=Relationship(source=source, target=target, edgeLabel=edge,
                                                       isNegated=negated))


EX = node(9, "?1", "existential")
BE = node(3, "be", "verb")
TRAFFIC = node(2, "traffic", "noun")
CENTRE = node(-1, "city centre", "LOC")
NEWCASTLE = node(6, "Newcastle", "GPE",
                 {("4", "in"), ("det", "the"), ("extra", (CENTRE,)), ("type", "stay in place")})
CTX = RewriteContext(node_functions=None, services=None, matchers=None)
RULE = ExistentialLocativeToHaveRule()


class TestPlainExistential(unittest.TestCase):
    def test_there_is_traffic_in_the_city_centre(self):
        k = kernel(BE, EX, TRAFFIC, {("SPACE", NEWCASTLE)})
        b = RULE.matches(k, CTX)
        self.assertIsNotNone(b)
        out = RULE.apply(k, b, CTX)
        rel = out.kernel
        self.assertEqual(rel.edgeLabel.named_entity, "have")
        self.assertEqual(rel.source.named_entity, "Newcastle")
        self.assertEqual(rel.target.named_entity, "traffic")
        self.assertEqual(dict(rel.source.properties), {"det": "the", "extra": (CENTRE,)})
        self.assertNotIn("SPACE", dict(out.properties))
        self.assertNotIn("CAUSATION", dict(out.properties))

    def test_negation_is_kept(self):
        k = kernel(BE, EX, TRAFFIC, {("SPACE", NEWCASTLE)}, negated=True)
        self.assertTrue(RULE.apply(k, RULE.matches(k, CTX), CTX).kernel.isNegated)

    def test_negated_space_left_alone(self):
        not_centre = SetOfSingletons(id=7, type=Grouping.NOT, entities=(NEWCASTLE,), min=0, max=1, confidence=1)
        self.assertIsNone(RULE.matches(kernel(BE, EX, TRAFFIC, {("SPACE", not_centre)}), CTX))

    def test_two_places_left_alone(self):
        terrace = node(8, "Prince Albert Terrace", "LOC")
        self.assertIsNone(RULE.matches(kernel(BE, EX, TRAFFIC, {("SPACE", (terrace, NEWCASTLE))}), CTX))

    def test_place_as_target_left_alone(self):
        self.assertIsNone(RULE.matches(kernel(BE, EX, node(5, "Leeds", "GPE"), {("SPACE", NEWCASTLE)}), CTX))

    def test_non_copula_left_alone(self):
        self.assertIsNone(RULE.matches(kernel(node(3, "flow", "verb"), EX, TRAFFIC, {("SPACE", NEWCASTLE)}), CTX))

    def test_specification_shape_unchanged(self):
        npcp = node(11, "NPCP", "LOC")
        reduction = node(12, "reduction", "noun")
        repairs = node(13, "repairs", "noun")
        k = kernel(BE, EX, repairs, {("SPACE", npcp), ("SPECIFICATION", reduction)})
        out = RULE.apply(k, RULE.matches(k, CTX), CTX)
        self.assertEqual((out.kernel.source.named_entity, out.kernel.target.named_entity), ("NPCP", "reduction"))
        self.assertEqual([v.named_entity for v in dict(out.properties)["CAUSATION"]], ["repairs"])



class _PropertyReadingMatcher:
    """Stands in for KernelOntologyMatchers.is_location_like, which tests the node's name and every property key and
    value against HOnK's location nouns; "in", "place" and "there" are among them."""
    PLACES = {"newcastle", "city centre", "leeds", "in", "place", "there"}

    def is_location_like(self, node):
        words = {str(node.named_entity).lower()}
        for k, v in dict(node.properties).items():
            for x in (k, *(v if isinstance(v, tuple) else (v,))):
                words |= set(str(getattr(x, "named_entity", x)).lower().split())
        return node.type in {"GPE", "LOC"} or bool(words & self.PLACES)


class TestWithPropertyReadingMatcher(unittest.TestCase):
    """The run's matcher (19 Nov 2026): the copula target carries a property such as "there", and every SPACE value
    "type: stay in place", so the property-reading test calls both places; the rule asks about their own names."""
    ctx = RewriteContext(node_functions=None, services=None, matchers=_PropertyReadingMatcher())

    def test_fires_on_the_run_shape(self):
        traffic = node(2, "traffic", "noun", {("expl", "there")})
        k = kernel(BE, EX, traffic, {("SPACE", NEWCASTLE)})
        self.assertTrue(self.ctx.matchers.is_location_like(traffic))
        rel = RULE.apply(k, RULE.matches(k, self.ctx), self.ctx).kernel
        self.assertEqual((rel.edgeLabel.named_entity, rel.source.named_entity, rel.target.named_entity),
                         ("have", "Newcastle", "traffic"))

    def test_place_named_target_still_left_alone(self):
        leeds = node(5, "Leeds", "noun")
        self.assertIsNone(RULE.matches(kernel(BE, EX, leeds, {("SPACE", NEWCASTLE)}), self.ctx))

    def test_adjective_target_left_alone(self):
        # "It is busy in Newcastle": the pronoun is an existential placeholder too, and the target an adjective.
        busy = node(2, "busy", "JJ")
        self.assertIsNone(RULE.matches(kernel(BE, EX, busy, {("SPACE", NEWCASTLE)}), self.ctx))

    def test_space_that_is_not_a_place_left_alone(self):
        morning = node(8, "morning", "noun", {("4", "in"), ("type", "stay in place")})
        self.assertIsNone(RULE.matches(kernel(BE, EX, TRAFFIC, {("SPACE", morning)}), self.ctx))


if __name__ == "__main__":
    unittest.main()
