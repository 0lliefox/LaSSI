"""Reproduce the result numbers of the CURB paper (ICNLSP 2026) from the released outputs.

    python reproduce_curb.py

Scores the released per-case comparison matrices (neet/curb_results/matrices) with eval_results.py, whose
label rules are those of the paper's Section 5.2, and checks every system's per-pair label against the frozen
record the paper's numbers were taken from (neet/curb_results/evaluation_output_compared.csv). Needs Python
3.10+ with pandas and scikit-learn; it does not run the LaSSI pipeline. See REPRODUCING.md for what the
released matrices are and why re-running the pipeline today does not regenerate them.
"""
import csv
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "neet", "curb_results")
LABELS = ("Supported", "Refuted", "Not Enough Evidence")


def macro_f1(gold, pred):
    scores = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores)


def per_system(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    systems = sorted(k[: -len("_pred")] for k in rows[0] if k.endswith("_pred"))
    out = {}
    for s in systems:
        gold = [r["label"].strip().title() for r in rows]
        pred = [str(r[s + "_pred"]).strip().title() for r in rows]
        out[s] = {"correct": sum(g == p for g, p in zip(gold, pred)), "n": len(rows),
                  "macro_f1": macro_f1(gold, pred), "labels": {r["item_id"]: p for r, p in zip(rows, pred)}}
    return out


def main():
    with tempfile.TemporaryDirectory(prefix="curb_repro_") as work:
        os.symlink(os.path.join(ROOT, "neet"), os.path.join(work, "neet"))
        os.symlink(os.path.join(ROOT, "LaSSI"), os.path.join(work, "LaSSI"))
        os.symlink(os.path.join(RESULTS, "matrices"), os.path.join(work, "catabolites"))
        os.makedirs(os.path.join(work, "results"))
        run = subprocess.run([sys.executable, os.path.join(ROOT, "eval_results.py")], cwd=work,
                             capture_output=True, text=True)
        if run.returncode != 0:
            sys.stderr.write(run.stdout + run.stderr)
            sys.exit("eval_results.py failed")
        scored = per_system(os.path.join(work, "evaluation_output_compared.csv"))
    frozen = per_system(os.path.join(RESULTS, "evaluation_output_compared.csv"))

    print(f"{'system':42s} {'from released matrices':>24s}   {'paper (frozen record)':>24s}   labels identical")
    for s in sorted(frozen):
        f = frozen[s]
        m = scored.get(s)
        left = f"{m['correct']}/{m['n']}  {100 * m['correct'] / m['n']:5.1f}%  F1 {m['macro_f1']:.3f}" if m else "-"
        right = f"{f['correct']}/{f['n']}  {100 * f['correct'] / f['n']:5.1f}%  F1 {f['macro_f1']:.3f}"
        same = m is not None and all(m["labels"].get(i) == p for i, p in f["labels"].items())
        print(f"{s:42s} {left:>24s}   {right:>24s}   {'yes' if same else 'NO'}")
    print("\nThe paper's figures are the right-hand column, computed from the frozen per-pair record. Re-scoring the"
          "\nreleased matrices reproduces every deterministic baseline exactly and LaSSI's 77/90. Two known exceptions"
          "\n(REPRODUCING.md): LaSSI's transport_007_S, wrong in both, is NEE in the record and Refuted in the released"
          "\nmatrix (Macro-F1 0.858 vs 0.857); and the LLM matrices were regenerated after submission, and LLM output"
          "\nis non-deterministic.")


if __name__ == "__main__":
    main()
