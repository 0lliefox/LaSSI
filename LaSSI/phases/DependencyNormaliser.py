__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""Dependency-level normalisation of the parser output (gsmDB.txt) before the graph grammar runs.

The grammar reads the dependency graphs written by the parser. A few constructions reach it in a shape whose
meaning the grammar cannot recover, because the link between a noun and what is predicated of it is lost on the
way. This step rewrites them into the shape of an equivalent construction that LaSSI already handles:

* judgement passive: "delays are considered unlikely" -> "delays are unlikely", when the complement of a past
  participle is a modal adjective (HOnK's ModalAdjective class). In a notice the one who judges is the issuing
  authority, as for the bare "Delays unlikely"; the modality itself is kept.
* reduced judgement relative: "delays considered unlikely" -> "unlikely delays" (the modal adjective becomes a
  modifier of the noun, which `KernelPostProcessor.promote_modal_modifiers` turns into MODALITY on that noun).
* passive purpose: "The road is closed to allow resurfacing" -> the "to" clause is a purpose clause (advcl), not
  the complement the parser gives it, when the participle's verb takes no clausal complement in WordNet.

The file format is kept exactly: `parse_gsm` and `serialise_gsm` round-trip byte for byte."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_BLOCK_SEPARATOR = "~~"


@dataclass
class GsmNode:
    id: str
    ell: List[str]
    xi: List[str]
    properties: List[Tuple[str, str]]          # raw (key, value) strings, without quotes
    phi: List[Tuple[str, List[Tuple[str, str]]]] = field(default_factory=list)  # (label, [(weight, target id)])

    def prop(self, key) -> Optional[str]:
        for k, v in self.properties:
            if k == key:
                return v
        return None

    def children(self, label=None):
        return [(lbl, tgt) for lbl, targets in self.phi for _, tgt in targets if label is None or lbl == label]


def _parse_node(text: str) -> GsmNode:
    lines = text.split("\n")
    assert lines[0].startswith("id:"), text[:80]
    node_id = lines[0][3:]
    i = 1

    def section(name):
        nonlocal i
        assert lines[i] == f"{name}:", (name, lines[i])
        i += 1
        values = []
        while lines[i] != ".":
            values.append(lines[i])
            i += 1
        i += 1
        return values

    ell = section("ell")
    xi = section("xi")
    raw_props = section("properties")
    properties = []
    for line in raw_props:
        key, value = line.split("\t", 1)
        properties.append((key[1:-1], value[1:-1]))
    assert lines[i] == "phi:", lines[i]
    phi_text = "\n".join(lines[i + 1:])
    phi = []
    if phi_text != ".":
        assert phi_text.endswith("\n;."), phi_text[-20:]
        for item in phi_text[:-3].split("\n;"):
            item_lines = item.split("\n")
            label = item_lines[0][1:-1]
            targets = []
            for t in item_lines[1:]:
                _, weight, target = t.split("\t")
                targets.append((weight, target))
            phi.append((label, targets))
    return GsmNode(node_id, ell, xi, properties, phi)


def _serialise_node(node: GsmNode) -> str:
    out = [f"id:{node.id}", "ell:", *node.ell, ".", "xi:", *node.xi, ".", "properties:"]
    out += [f'"{k}"\t"{v}"' for k, v in node.properties]
    out += [".", "phi:"]
    if not node.phi:
        out.append(".")
    else:
        items = []
        for label, targets in node.phi:
            items.append("\n".join([f'"{label}"'] + [f"\t{w}\t{t}" for w, t in targets]))
        out.append("\n;".join(items) + "\n;.")
    return "\n".join(out)


def parse_gsm(text: str):
    """Returns a list of blocks; each block is (leading, [GsmNode], trailing) whitespace-preserving."""
    blocks = []
    for raw in text.split(_BLOCK_SEPARATOR):
        stripped = raw.strip("\n")
        leading = raw[:len(raw) - len(raw.lstrip("\n"))]
        trailing = raw[len(raw.rstrip("\n")):]
        nodes = [_parse_node(chunk) for chunk in stripped.split("\n\n")] if stripped else []
        blocks.append((leading, nodes, trailing))
    return blocks


def serialise_gsm(blocks) -> str:
    return _BLOCK_SEPARATOR.join(
        leading + "\n\n".join(_serialise_node(n) for n in nodes) + trailing for leading, nodes, trailing in blocks)


_PASSIVE_SUBJECT = "nsubjpass"
_PASSIVE_AUXILIARY = "auxpass"
_COMPLEMENT = "xcomp"
_OBJECTS = frozenset({"obj", "dobj"})


def _is_past_participle(node: GsmNode) -> bool:
    return bool(node.ell) and node.ell[0] == "verb" and node.prop("xpos") == "VBN"


def _modal_complement(nodes_by_id, verb: GsmNode, modal_adjectives):
    """The modal-adjective complement of `verb` ("considered *unlikely*"), or None."""
    for label, target in verb.children(_COMPLEMENT):
        adjective = nodes_by_id.get(target)
        if (adjective is not None and adjective.ell and adjective.ell[0] == "JJ" and adjective.xi and
                adjective.xi[0].lower() in modal_adjectives):
            return adjective
    return None


def _add_edge(node: GsmNode, label: str, weight: str, target: str):
    for lbl, targets in node.phi:
        if lbl == label:
            if not any(t == target for _, t in targets):
                targets.append((weight, target))
            return
    node.phi.append((label, [(weight, target)]))


def _remove_edge(node: GsmNode, label: str, target: str):
    node.phi = [(lbl, [(w, t) for w, t in targets if not (lbl == label and t == target)]) for lbl, targets in node.phi]
    node.phi = [(lbl, targets) for lbl, targets in node.phi if targets]


def _redirect_incoming(nodes, old_id: str, new_id: str):
    for node in nodes:
        node.phi = [(lbl, [(w, new_id if t == old_id else t) for w, t in targets]) for lbl, targets in node.phi]


def _governor(nodes, node_id: str) -> Optional[GsmNode]:
    for node in nodes:
        if any(t == node_id for _, t in node.children()):
            return node
    return None


def _collapse_judgement_passive(nodes, modal_adjectives) -> bool:
    """"delays are considered unlikely" -> "delays are unlikely": the modal adjective heads the clause, with the
    passive subject as its subject and the passive auxiliary as its copula; it inherits the participle's other
    dependents and incoming edge."""
    nodes_by_id = {n.id: n for n in nodes}
    for verb in nodes:
        if not _is_past_participle(verb) or not verb.children(_PASSIVE_SUBJECT):
            continue
        adjective = _modal_complement(nodes_by_id, verb, modal_adjectives)
        if adjective is None:
            continue
        for label, targets in verb.phi:
            for weight, target in targets:
                if label == _COMPLEMENT and target == adjective.id:
                    continue
                new_label = {_PASSIVE_SUBJECT: "nsubj", _PASSIVE_AUXILIARY: "cop"}.get(label, label)
                _add_edge(adjective, new_label, weight, target)
        if "root" in verb.ell and "root" not in adjective.ell:
            adjective.ell.append("root")
        _redirect_incoming(nodes, verb.id, adjective.id)
        _remove_node(nodes, verb)
        return True
    return False


def _collapse_judgement_relative(nodes, modal_adjectives) -> bool:
    """"delays considered unlikely" -> "unlikely delays": a reduced relative whose past participle has no subject or
    object of its own and a modal-adjective complement becomes a modifier of the noun; the participle's other
    dependents (e.g. "before work ends") move to the noun's governor."""
    nodes_by_id = {n.id: n for n in nodes}
    for noun in nodes:
        for label, target in noun.children("acl"):
            verb = nodes_by_id.get(target)
            if verb is None or not _is_past_participle(verb):
                continue
            if verb.children(_PASSIVE_SUBJECT) or any(verb.children(o) for o in _OBJECTS):
                continue
            adjective = _modal_complement(nodes_by_id, verb, modal_adjectives)
            if adjective is None:
                continue
            weight = next(w for lbl, targets in noun.phi if lbl == "acl" for w, t in targets if t == verb.id)
            _remove_edge(noun, "acl", verb.id)
            _add_edge(noun, "amod", weight, adjective.id)
            host = _governor(nodes, noun.id) or noun
            for v_label, v_targets in verb.phi:
                for v_weight, v_target in v_targets:
                    if v_label == _COMPLEMENT and v_target == adjective.id:
                        continue
                    _add_edge(host, v_label, v_weight, v_target)
            _remove_node(nodes, verb)
            return True
    return False


# WordNet verb frames with a clausal complement: the infinitive frames 24 "Somebody ----s somebody to INFINITIVE",
# 25 "... somebody INFINITIVE", 28 "Somebody ----s to INFINITIVE", 29 "... whether INFINITIVE", 32 "Somebody ----s
# INFINITIVE", 35 "Something ----s INFINITIVE", and 26 "Somebody ----s that CLAUSE". A passive "X is V-ed to Y" with Y
# a complement needs one of them: "expected to" (24, 28), "forecast to" and "believed to" (26, the raising passive of
# verbs of saying, thinking and predicting).
_CLAUSAL_COMPLEMENT_FRAMES = frozenset({24, 25, 26, 28, 29, 32, 35})


def _takes_infinitive_complement(lemma: str) -> Optional[bool]:
    """Whether WordNet lists a clausal-complement frame (`_CLAUSAL_COMPLEMENT_FRAMES`) for the verb `lemma`; None if
    WordNet has no verb sense."""
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(lemma, pos=wn.VERB)
    if not synsets:
        return None
    return any(set(l.frame_ids()) & _CLAUSAL_COMPLEMENT_FRAMES
               for s in synsets for l in s.lemmas() if l.name().lower() == lemma.lower())


def _reattach_passive_purpose(nodes) -> bool:
    """"The road is closed to allow resurfacing": the parser attaches the purpose clause as a complement (xcomp),
    the same structure as "Trains are expected to run normally". A passive participle whose verb takes no clausal
    complement in WordNet ("close") cannot have one, so its "to" clause is a purpose clause (advcl); verbs that do
    ("expect", "plan", "forecast", "believe") keep the complement reading."""
    nodes_by_id = {n.id: n for n in nodes}
    for verb in nodes:
        if not _is_past_participle(verb) or not (verb.children(_PASSIVE_SUBJECT) or verb.children(_PASSIVE_AUXILIARY)):
            continue
        lemma = verb.prop("lemma") or (verb.xi[0] if verb.xi else "")
        for label, target in verb.children(_COMPLEMENT):
            clause = nodes_by_id.get(target)
            if clause is None or not clause.ell or clause.ell[0] != "verb":
                continue
            if not any((nodes_by_id.get(m) is not None and nodes_by_id[m].ell[:1] == ["TO"])
                       for _, m in clause.children("mark")):
                continue
            if _takes_infinitive_complement(lemma) is not False:
                continue
            weight = next(w for lbl, targets in verb.phi if lbl == _COMPLEMENT for w, t in targets if t == target)
            _remove_edge(verb, _COMPLEMENT, target)
            _add_edge(verb, "advcl", weight, target)
            return True
    return False


def _remove_node(nodes, removed: GsmNode):
    """Remove `removed`, keeping node ids contiguous (the grammar engine indexes nodes by id): the node with the
    highest id takes the freed id."""
    nodes.remove(removed)
    if not nodes:
        return
    last = max(nodes, key=lambda n: int(n.id))
    if int(last.id) > int(removed.id):
        old_id = last.id
        last.id = removed.id
        _redirect_incoming(nodes, old_id, removed.id)


def normalise_gsm(text: str, modal_adjectives) -> str:
    """Apply the rewrites above to every graph in a gsmDB text."""
    modal_adjectives = {m.lower() for m in modal_adjectives or ()}
    if not modal_adjectives:
        return text
    blocks = parse_gsm(text)
    for _, nodes, _ in blocks:
        while (_collapse_judgement_passive(nodes, modal_adjectives) or
               _collapse_judgement_relative(nodes, modal_adjectives) or
               _reattach_passive_purpose(nodes)):
            pass
    return serialise_gsm(blocks)
