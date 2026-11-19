"""The honk table (label -> generalised types, for the fuzzy matcher) built by streaming HOnK's N-Triples file.

v4.4 made it with one SPARQL `SELECT DISTINCT ?s ?src_label ?dst { ?s a ?dst ; rdfs:label ?src_label }` and a Python
dict over every typed node (HOnK.getTypedObjects): on HOnK v2.0.0, 20M type and 24M label triples at once. Here the
type and label triples are written out by subject, sorted on disk (`sort`, C locale), joined, and grouped by label, so
memory stays small whatever the graph's size. Each label's types are the same as v4.4's rule gives: every rdf:type of
every node with that label, str(type)[len(honk#):], generalised one at a time by HOnK.most_specific_type, "None" dropped
when the label has another type. On 1% and 10% slices of HOnK v2.0.0 the label -> type sets equal v4.4's dump.

Order of a label's types (Oliver, 19 Nov: follow v4.4's table). v4.4 joined a Python set, so the same set appears in
different orders (JJ,noun and noun,JJ), and LaSSI takes the first of equal-score candidates. Given v4.4's table as a
reference (`label<TAB>types`, COPY text format, e.g. `\\copy (select idx, type from honk) to ...`): a label whose type
set is unchanged keeps v4.4's string; otherwise the order v4.4's table uses most for that set; otherwise PRECEDENCE.

usage: python -m LaSSI.HOnK.honk_table <graph.nt> <out.tab> [--reference <v44_table.tsv>] [--work <dir>]
"""
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

HONK_NS = "https://ofox.co.uk/honk#"
RDF_TYPE = b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
RDFS_LABEL = b"<http://www.w3.org/2000/01/rdf-schema#label>"
# most_specific_type's own ranking, for a type set v4.4's table never had.
PRECEDENCE = ("verb", "GPE", "LOC", "FAC", "ORG", "noun", "ENTITY", "JJ", "RB", "IN", "None")

_ESC = re.compile(r'\\(?:u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|(.))', re.S)
_SIMPLE = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\"}


def _unescape(lexical):
    return _ESC.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)) if (m.group(1) or m.group(2))
                    else _SIMPLE[m.group(3)], lexical)


def term_value(token: bytes) -> str:
    """str() of an N-Triples term as LaSSI reads it from the store: an IRI's text, a blank node's id, a literal's
    lexical form."""
    if token.startswith(b"<"):
        return token[1:-1].decode("utf-8")
    if token.startswith(b"_:"):
        return token[2:].decode("utf-8")
    text = token.decode("utf-8")
    return _unescape(text[1:text.rfind('"')])


def copy_escape(text: str) -> str:
    """A value in PostgreSQL's COPY text format (what FuzzyStringMatchDatabase.create loads)."""
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def _sort(path, work, mem="1G"):
    """Sort a `key<TAB>value` file by key, bytewise. Not -u: with -k it would keep one line per key; repeated lines
    are harmless, since every reader collects sets."""
    subprocess.run(["sort", "-t", "\t", "-k1,1", "-S", mem, "-T", work, "-o", path, path], check=True,
                   env=dict(os.environ, LC_ALL="C"))


def _groups(fh):
    """(key, [values]) runs of a file of `key<TAB>value` lines sorted by key."""
    key, vals = None, []
    for line in fh:
        k, v = line.rstrip(b"\n").split(b"\t", 1)
        if k != key:
            if key is not None:
                yield key, vals
            key, vals = k, []
        vals.append(v)
    if key is not None:
        yield key, vals


def label_type_sets(nt_path, work, most_specific_type):
    """Yield (label in COPY text format, frozenset of generalised types) for every labelled, typed node's label,
    in label order (bytes, C locale)."""
    types_f, labels_f, pairs_f = (os.path.join(work, n) for n in ("types.tsv", "labels.tsv", "label_type.tsv"))
    generalised = {}
    with open(nt_path, "rb") as fh, open(types_f, "wb") as ft, open(labels_f, "wb") as fl:
        for line in fh:
            s, p, rest = line.split(b" ", 2)
            if p != RDF_TYPE and p != RDFS_LABEL:
                continue
            o = rest.rstrip(b"\r\n")
            o = o[:-1].rstrip() if o.endswith(b".") else o
            if p == RDF_TYPE:
                g = generalised.get(o)
                if g is None:
                    g = generalised[o] = most_specific_type([term_value(o)[len(HONK_NS):]]).encode()
                ft.write(s + b"\t" + g + b"\n")
            else:
                fl.write(s + b"\t" + copy_escape(term_value(o)).encode("utf-8") + b"\n")
    for f in (types_f, labels_f):
        _sort(f, work)
    with open(types_f, "rb") as ft, open(labels_f, "rb") as fl, open(pairs_f, "wb") as fp:
        gt, gl = _groups(ft), _groups(fl)
        a, b = next(gt, None), next(gl, None)
        while a is not None and b is not None:
            if a[0] < b[0]:
                a = next(gt, None)
            elif a[0] > b[0]:
                b = next(gl, None)
            else:
                for label in b[1]:
                    for t in a[1]:
                        fp.write(label + b"\t" + t + b"\n")
                a, b = next(gt, None), next(gl, None)
    os.remove(types_f), os.remove(labels_f)
    _sort(pairs_f, work)
    with open(pairs_f, "rb") as fp:
        for label, ts in _groups(fp):
            gen = {t.decode() for t in ts}
            if len(gen) > 1:
                gen.discard("None")
            yield label.decode("utf-8"), frozenset(gen)
    os.remove(pairs_f)


class TypeOrder:
    """The order of a label's types in the table, following a reference table (v4.4's), else PRECEDENCE."""

    def __init__(self, reference_path=None, work=None):
        self.reference_path, self.by_set = None, {}
        if reference_path:
            self.reference_path = os.path.join(work, "reference.tsv")
            counts = defaultdict(Counter)
            with open(reference_path, "rb") as src, open(self.reference_path, "wb") as dst:
                for line in src:
                    label, types = line.rstrip(b"\n").split(b"\t", 1)
                    counts[frozenset(types.decode().split(","))][types.decode()] += 1
                    dst.write(line if line.endswith(b"\n") else line + b"\n")
            _sort(self.reference_path, work)
            # the most frequent string of each set; equal counts: the smallest string, for a fixed result
            self.by_set = {s: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for s, c in counts.items()}

    def default(self, types):
        return self.by_set.get(types) or ",".join(sorted(types, key=lambda t: (PRECEDENCE.index(t)
                                                                             if t in PRECEDENCE else len(PRECEDENCE), t)))

    def ordered(self, rows):
        """(label, type string) for each (label, set) of rows (in label order)."""
        if self.reference_path is None:
            for label, types in rows:
                yield label, self.default(types)
            return
        with open(self.reference_path, "rb") as fh:
            ref = ((k.decode("utf-8"), v[-1].decode()) for k, v in _groups(fh))
            r = next(ref, None)
            for label, types in rows:
                key = label.encode("utf-8")
                while r is not None and r[0].encode("utf-8") < key:
                    r = next(ref, None)
                if r is not None and r[0] == label and frozenset(r[1].split(",")) == types:
                    yield label, r[1]
                else:
                    yield label, self.default(types)


def write_tab(nt_path, out, most_specific_type, reference_path=None, work=None):
    """Write the table file FuzzyStringMatchDatabase.create loads (`id idx t type`, a header line first) to *out* (a
    path or a text file). Returns the number of rows."""
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        order = TypeOrder(reference_path, tmp)
        f = out if hasattr(out, "write") else open(out, "w", encoding="utf-8")
        try:
            f.write(f"id\tidx\tt\ttype{os.linesep}")
            n = 0
            for n, (label, types) in enumerate(order.ordered(label_type_sets(nt_path, tmp, most_specific_type)), 1):
                f.write(f"{n}\t{label}\t{label}\t{types}{os.linesep}")
        finally:
            if f is not out:
                f.close()
    return n


if __name__ == "__main__":
    import argparse
    from LaSSI.HOnK.HOnK import HOnK
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph"), ap.add_argument("out")
    ap.add_argument("--reference"), ap.add_argument("--work")
    a = ap.parse_args()
    rows = write_tab(a.graph, a.out, HOnK.most_specific_type, a.reference, a.work)
    print(f"{rows:,} rows -> {a.out}", file=sys.stderr)
