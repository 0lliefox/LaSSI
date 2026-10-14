"""Which cases' time values change when relative expressions are resolved against a reference date (issue 158).

    python eval/anchor_time_check.py [--parity] [case ...]

For every sub-sentence the NER stage saw (the case's cached meuDBs.json), compares the Java wrapper's time units
(anchored to today) with LaSSI.external_services.AnchoredTime's, anchored to the case's reference date (its sidecar
`neet/evidence_cases/<case>.meta.yaml`). `--parity` anchors to today instead, so the two must agree exactly: that
checks the anchored pipeline reproduces the wrapper. Needs the JVM, not HOnK; writes nothing.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.getcwd())

CASES_DIR = "neet/evidence_cases"


def sub_sentences(case):
    with open(f"catabolites/{case}/meuDBs.json") as fh:
        return [row["first_sentence"] for row in json.load(fh)]


def comparable(units):
    return [(u.get("text"), u.get("type"), u.get("start_char"), u.get("end_char")) for u in units]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    parity = "--parity" in sys.argv
    cases = args or sorted(f[:-5] for f in os.listdir(CASES_DIR) if f.endswith(".yaml") and not f.endswith(".meta.yaml"))
    from LaSSI.LaSSI import resolve_reference_date
    from LaSSI.external_services.AnchoredTime import time_units
    from StanfordNLPExtractor.OldWrapper import OldWrapper
    wrapper = OldWrapper.getInstance()
    changed = []
    for case in cases:
        if not os.path.isfile(f"catabolites/{case}/meuDBs.json"):
            print(f"{case}: no cached meuDB, skipped")
            continue
        day = datetime.date.today().isoformat() if parity else resolve_reference_date(f"{CASES_DIR}/{case}.yaml")
        if day is None:
            print(f"{case}: no reference date")
            continue
        sentences = sub_sentences(case)
        before = wrapper.getTimeUnits(sentences)
        after = time_units(sentences, day)
        diffs = [(i, comparable(b), comparable(a)) for i, (b, a) in enumerate(zip(before, after)) if comparable(b) != comparable(a)]
        print(f"{case} ({day}): {'identical' if not diffs else f'{len(diffs)} sentence(s) differ'}")
        for i, b, a in diffs:
            print(f"   [{i}] {sentences[i][:90]!r}\n       wrapper: {b}\n       anchored: {a}")
        if diffs:
            changed.append(case)
    print("CHANGED:", " ".join(changed))


if __name__ == "__main__":
    main()
