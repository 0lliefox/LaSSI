# <img src="extra/lassiLogo.svg" style="height:80px; width: auto;" alt="Logo: Credits to Oliver Robert Fox (2024)" /> LaSSI

LaSSI (`LogicAl, Structural, and Semantic text Interpretation`) reads natural language through symbolic steps that can
each be checked. It turns text into Extended First-Order Logic (eFOL) formulae and scores how far one sentence implies
another by BDD model counting, so every decision can be traced. The pipeline is domain-general; the current experiments
use **NEET**, a corpus of transport, roadworks, crime and weather notices.

## Status

This is **LaSSI v4.5**, frozen at tag `v4.5` on 27 November 2026 for the evaluation on CURB. It reads
[HOnK v1.6.1](https://github.com/0lliefox/HOnK/tree/v1.6.1). The previous version is branch `v4.4`.

Against v4.4, v4.5:

- reads HOnK v1.6.1 from its release N-Triples, with a table and cache of its own;
- lets only a different town contradict a place (another street in the same town, or a forecast for another town,
  is not a contradiction);
- reads a clock time stated in UTC as the date's own time;
- gives no score to a sentence it cannot parse;
- can run the LLM baselines from their released weights through Hugging Face.

Each change has unit tests in `LaSSI/tests/`.

## Pipeline

Each phase caches its results per sentence in `catabolites/`.

| Phase | Module (`LaSSI/phases/`) | Does |
|---|---|---|
| SentenceLoader | `SentenceLoader.py` | Reads the YAML rows |
| StructuredSentenceLoader | `StructuredSentenceLoader.py` | Splits rows into chunks; protects timestamps, date ranges and metrics |
| RowChunkPipeline | `RowChunkPipeline.py` | Classifies each chunk's role and merges the chunks' kernels back into the row |
| ExplainTextWithNER | `ResolveBasicTypes.py` | Stanza tagging and parsing; matches words against HOnK (meuDB) |
| GetGSMString | `GetGSMString.py` | Stanford CoreNLP (Java) gives the Generalised Semistructured Model (GSM) |
| ApplyGraphGrammars | `ApplyGraphGrammars.py` | DatagramDB graph-grammar rules |
| SemanticGraphRewriting | `SemanticGraphRewriting.py` | GSM to an entity-relationship graph, with structural rewrites |
| LogicalRewriting | `LogicalRewriting.py` | eFOL formulae |
| CalculateMatrix | `CalculateMatrix.py` | The sentence-to-sentence similarity matrix |

The structural rewrites are 30 rules (14 declarative, 16 in Python), all registered in `raw_data/structural_rewrites.json`.

## Knowledge

- **HOnK v1.6.1**: `LaSSI/HOnK-v1-6-1.nt` is a link to the release's N-Triples file and is not in the repository.
  LaSSI builds a store (`cache_v161`) and a Postgres table (`honk_v161`) from it, and rebuilds both when the file
  changes. The names are set in `LaSSI/Configuration.py`.
- `LaSSI/LifecycleStates.ttl`: contradictions between states (e.g. *reduction in spaces* ⊥ *operating normally*).
- `LaSSI/Paraphrase.ttl`: predicate equivalences (e.g. *operate* ≡ *run*).

## Similarity backends

Chosen by the `transformer` string:

| Prefix | Backend | Example |
|---|---|---|
| *(none)* | eFOL BDD model counting (`Logical`, the default) | |
| bare name | Sentence-transformer embeddings | `all-MiniLM-L6-v2` |
| `NLI#` | NLI cross-encoder | `NLI#cross-encoder/nli-MiniLM2-L6-H768` |
| `LLM#` | LLM through Ollama, or the released weights with `LASSI_LLM_BACKEND=hf` and `LASSI_LLM_HF_MODELS` | `LLM#qwen2.5:7b` |
| `LLMHOnK#` | LLM given HOnK's facts as evidence | `LLMHOnK#qwen2.5:7b#all` |
| `RAG#` | ColBERT retrieval | `RAG#colbert-ir/colbertv2.0` |

## Installing

```bash
pip install requests==2.31.0 setuptools==66.1.1
sudo apt-get install libpq-dev   # macOS: brew install libpq
pip install .
```

Java is needed for Stanford CoreNLP. LaSSI needs a PostgreSQL database and user, as named in `connection.yaml`
(database `lassi_v45`, user `lassi`):

```postgresql
create database lassi_v45;
create user lassi with encrypted password 'drowssap';
grant all privileges on database lassi_v45 to lassi;
\c lassi_v45 postgres
grant all on schema public to lassi;
```

## Running

```bash
python main.py                                 # default dataset and config
python main.py <dataset.yaml> <connection.yaml>
```

`python visualiser.py`, from the repository root, serves the LaSSI Visualiser (FastAPI and Dash) over `catabolites/`
on port 8000.

`SentenceRepresentation` in `LaSSI/Configuration.py` chooses the form: `Logical` (the default), `LogicalGraph`,
`SimpleGraph`, `FullText`, and each graph or logical form with `DisabledAPriori`.

## Tests

```bash
python -m unittest LaSSI.tests.test_v45_honk_v2        # one of the unit tests (hand-built inputs)
zsh regen_gold.sh                                       # rebuilds the 20 gold cases' outputs (one process each)
python -m unittest LaSSI/tests/test_similarities.py     # compares them with the gold matrices
```

`test_similarities.py` compares each case's matrix with `LaSSI/tests/assertions/similarities_neet.json`. A gold
`"null"` accepts any score strictly between 0 and 1; any other value must match exactly.

## Data and reproduction

NEET's annotated data is on [OSF](https://osf.io/fu4bg/overview?view_only=c321084ba74c488fb587fc4b54cc290f). Test
sentences are in `test_sentences/`. `REPRODUCING.md` recomputes the CURB results of the ICNLSP 2026 paper from the
released matrices; those matrices came from an earlier version of LaSSI, not v4.5.

## Authors

* Oliver Robert Fox (2023 –)
* Giacomo Bergami (2020 – 2025)
* Franco O. Saez Vander Linder (2025)
