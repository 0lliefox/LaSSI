# Reproducing the CURB results (ICNLSP 2026)

This branch is the codebase linked from *CURB: Civic Urban Reasoning Benchmark* (Fox and Solaiman, ICNLSP 2026).
The paper's data and outputs are also on OSF: <https://osf.io/fu4bg/>.

## One command

```
python reproduce_curb.py
```

This needs Python 3.10+ with `pandas` and `scikit-learn`; it does not run the LaSSI pipeline. It does three things:

1. It scores the released per-case comparison matrices (`neet/curb_results/matrices/`, the same files as on OSF)
   with `eval_results.py`, using the label rules of the paper's Section 5.2.
2. It recomputes accuracy and Macro-F1 for every system from the frozen per-pair record that the paper's numbers
   were taken from (`neet/curb_results/evaluation_output_compared.csv`).
3. It checks the two against each other, label by label.

Expected outcome:

| System | From the released matrices | Paper (frozen record) |
|---|---|---|
| LaSSI (`Logical`) | 77/90, Macro-F1 0.857 | 77/90, Macro-F1 0.858 |
| `nli-MiniLM2-L6-H768` | 75/90, identical labels | 75/90 |
| `nli-deberta-v3-base` | 73/90, identical labels | 73/90 |
| `bge-reranker-v2-m3` | 51/90, identical labels | 51/90 |
| `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2` | 43, 41, 40/90, identical labels | 43, 41, 40/90 |
| `gemma4:e2b`, `llama3.2:3b`, `qwen2.5:7b` | 61, 54, 62/90 | 64, 57, 59/90 |

There are two known differences, both recorded in the OSF README:

- **LaSSI's `transport_007_S`** is wrong in both. The frozen record (8 June 2026) has 0.5, which reads as Not Enough
  Evidence. The released matrix (10 June 2026) has 0.0, which reads as Refuted. The accuracy is identical, and
  Macro-F1 moves in the third decimal place.
- **The LLM matrices** were regenerated after submission (seeding was added to the harness later), and LLM
  generation is not deterministic. The paper's LLM figures are the frozen record.

The CSV in `neet/neet_v1.csv` differs from the OSF release only in the wording of one claim (`transport_004_N`).
Labels, domains and claim types are identical, so scoring is unaffected.

## How LaSSI is scored

LaSSI's label for each pair comes from its own matrix, in both directions: 0.0 in either direction is Refuted,
1.0 in either direction is Supported, and otherwise the pair is Not Enough Evidence. The curated matrices in
`LaSSI/tests/assertions/similarities_neet.json` are part of the benchmark definition. They are used only to
identify the pre-specified asymmetric cases (a Supported claim whose curated evidence-to-claim value is itself
partial, such as "on or near X" against "near X"). For those cases, a mutually partial computed pair is resolved
as support. This is the rule in Section 5.2 of the paper, and the one `get_label_matrix_output` applies to the
LLM matrices.

Until this commit, `eval_results.py` took LaSSI's labels for the 20 curated cases from the curated cells
themselves rather than from LaSSI's output. On the released matrices both versions give the same labels and the
same 77/90. The corrected version scores what LaSSI actually produced.

## Why the matrices are released rather than regenerated

The LaSSI matrices in the paper were produced on 10 June 2026 for the original submission. The camera-ready (4
August 2026) reports the same results. Re-running this branch today does not regenerate those matrices, for the
reasons below. Each was measured by re-running the pipeline with one factor changed at a time.

1. **Synonymy.** This branch bounds the synonymy closure (commit `4b8bcf7`, 4 July 2026), which was added after the
   evaluation. With the bound, the branch scores about 60/90 on CURB with HOnK v1.5.0. Without it, the score is 72/90 (hash
   seed 2).
2. **HOnK build.** The evaluation used a HOnK build from before v1.5.0 that has not been preserved. On the code
   that produced the matrices (`8fee848`), the surviving 11 May 2026 build reproduces 26 of the 30 published
   matrices, and v1.5.0 reproduces the other 4. HOnK v1.5.0 includes two fixes from 7 July 2026 that change
   LaSSI's behaviour:
   - **Proper names.** Wiktionary's `name` category was previously mapped to `Pronoun`, which put about 196,000
     proper names, and through them common words such as "rain" and "sunny", into LaSSI's pronoun lexicon.
     `transport_002_S` depended on this.
   - **Word senses.** Concepts are now keyed by sense, so the bare word "forecast" no longer carries the
     transitive-verb type it previously inherited. LaSSI drops a verb's object unless the verb is listed as
     transitive (`LaSSI/structures/kernels/SentenceX.py`). Under v1.5.0, "Newcastle is forecast to have a sunny day…"
     therefore loses its object, and five weather claims are no longer supported. Adding "forecast" alone to
     v1.5.0's transitive-verb list restores `weather_001` exactly, and the 11 May build's list restores all five.
3. **Run-to-run variation.** Parts of the pipeline iterate over sets of strings, whose order depends on Python's
   per-process hash seed (`PYTHONHASHSEED`). Different seeds move a small number of items.

The released matrices and the frozen record are therefore the authoritative outputs of the evaluated system, and
`reproduce_curb.py` recomputes every reported number from them.
