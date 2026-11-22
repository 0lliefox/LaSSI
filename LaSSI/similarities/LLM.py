import json
import os
import re
import threading
import httpx
from openai import OpenAI

# Per-request timeout for Ollama inference. 120s is generous for local models;
# the openai default (600s × 3 attempts) would block runs for 30+ minutes on timeout.
_OLLAMA_TIMEOUT = httpx.Timeout(timeout=120.0, connect=5.0)

# The baselines' backend (evaluation plan, section 4.5, 19 Nov 2026), chosen by LASSI_LLM_BACKEND:
#   "ollama" (default, unchanged): an Ollama daemon's quantised build of the model, temperature 0.1, seed 42.
#   "hf": the model's own released weights through Hugging Face transformers, in the dtype it was released in, at a
#         pinned revision, decoding greedily (temperature 0), with the same prompt. LASSI_LLM_HF_MODELS names a JSON
#         file mapping each model name as LaSSI knows it ("qwen2.5:7b") to {"repo": ..., "revision": ...}. Weights are
#         never downloaded during a run: a model missing from the local Hugging Face cache stops the run. A response
#         that gives no score returns None (no usable output, plan section 4.6), never a default value.
_HF_MAX_NEW_TOKENS = 512
_HF_LOADED = {}
_HF_LOCK = threading.Lock()


def _llm_backend():
    backend = os.environ.get("LASSI_LLM_BACKEND", "ollama")
    if backend not in ("ollama", "hf"):
        raise ValueError(f"LASSI_LLM_BACKEND={backend!r}: expected 'ollama' or 'hf'")
    return backend


def _hf_spec(model_name):
    path = os.environ.get("LASSI_LLM_HF_MODELS")
    if not path:
        raise RuntimeError("LASSI_LLM_BACKEND=hf needs LASSI_LLM_HF_MODELS, a JSON file of {model: {repo, revision}}")
    with open(path) as f:
        specs = json.load(f)
    spec = specs.get(model_name)
    if not spec or not spec.get("repo") or not spec.get("revision"):
        raise RuntimeError(f"{path}: no repo and pinned revision for {model_name!r}")
    return spec


def _hf_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hf_load(model_name):
    """(tokenizer, model, lock) for model_name, loaded once from the local cache at its pinned revision."""
    with _HF_LOCK:
        if model_name not in _HF_LOADED:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            spec = _hf_spec(model_name)
            kw = {"revision": spec["revision"], "local_files_only": True}
            tok = AutoTokenizer.from_pretrained(spec["repo"], **kw)
            try:
                model = AutoModelForCausalLM.from_pretrained(spec["repo"], torch_dtype="auto", **kw)
            except ValueError:
                # Multimodal releases (Gemma 4: Gemma4ForConditionalGeneration) register under image-text-to-text;
                # only their text input is used here.
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(spec["repo"], torch_dtype="auto", **kw)
            model.to(_hf_device()).eval()
            _HF_LOADED[model_name] = (tok, model, threading.Lock())
        return _HF_LOADED[model_name]


def _parse_score(text):
    """The implication_score from a response: the response itself as JSON, else the first JSON object in it that
    carries the key. Raises ValueError when there is none."""
    candidates = [text.strip()] + re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "implication_score" in data:
            return {"implication_score": float(data["implication_score"]), "reasoning": data.get("reasoning", "")}
    raise ValueError("no JSON object with 'implication_score' in the response")


class NoScoreAtAll(RuntimeError):
    """Every computed cell of a case gave no usable score: a fault in the set-up, not the model's answer."""


def no_score_counts(matrix, row0_only=False):
    """(computed, without a score) over a matrix's off-diagonal cells, the ones the model was asked. In row-0 mode only
    row 0 is asked; the other rows hold None without a call."""
    computed = none = 0
    for i, row in enumerate(matrix):
        if row0_only and i > 0:
            continue
        for j, value in enumerate(row):
            if i != j:
                computed += 1
                none += value is None
    return computed, none


def check_no_score(model_name, computed, none):
    """The line to log for a case's matrix. A pair with no usable output counts as wrong (plan section 4.6), so a case
    where every computed cell has none would read as a very bad model; it stops the run instead (Oliver, 21 Nov:
    "Stop if a case gets no score at all")."""
    line = f"[LLM] {model_name}: {none} of {computed} computed cells gave no usable score"
    if computed and none == computed:
        raise NoScoreAtAll(line + ": every one, so the set-up is at fault, not the model; stopping")
    return line


class LLMPrompt:
    def __init__(self, model_name="qwen2.5"):
        self.model_name = model_name
        self.backend = _llm_backend()
        if self.backend == "hf":
            _hf_load(model_name)
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=_OLLAMA_TIMEOUT,
            max_retries=0,
        )

    _PROMPT_TEMPLATE = (
        "You are a strict logical reasoning assistant.\n\n"
        "Premise: '{premise}'\n"
        "Consequence: '{consequence}'\n\n"
        "Analyze if the premise implies the consequence. "
        "Respond ONLY in valid JSON using the following format:\n"
        "{{\n"
        "  \"reasoning\": \"Briefly explain your logic here\",\n"
        "  \"implication_score\": 1.0 (if fully true), 0.5 (if partially true), or 0.0 (if false)\n"
        "}}"
    )

    def _build_prompt(self, premise: str, consequence: str) -> str:
        """Build the prompt sent to the model. Subclasses override this to
        inject extra grounding (e.g. HOnK ontology facts) while reusing the
        Ollama dispatch/parse plumbing in `_dispatch`."""
        return self._PROMPT_TEMPLATE.format(premise=premise, consequence=consequence)

    def _query(self, premise: str, consequence: str) -> dict:
        """Send one prompt to Ollama and return the parsed JSON dict."""
        return self._dispatch(self._build_prompt(premise, consequence))

    def _dispatch_hf(self, prompt: str) -> dict:
        """Greedy decoding is deterministic, so a failed response is not retried: it returns None as the score.
        Only a response without a usable score gives None; a failure to generate one (a fault in the set-up) is raised,
        never counted as the model's answer."""
        import torch
        tok, model, lock = _hf_load(self.model_name)
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt")
        # transformers 4.x returns the ids tensor; 5.x a BatchEncoding with input_ids and attention_mask
        if isinstance(enc, torch.Tensor):
            ids, mask = enc, torch.ones_like(enc)
        else:
            ids = enc["input_ids"]
            mask = enc["attention_mask"] if "attention_mask" in enc else torch.ones_like(ids)
        ids, mask = ids.to(model.device), mask.to(model.device)
        with lock, torch.no_grad():
            out = model.generate(ids, attention_mask=mask, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, max_new_tokens=_HF_MAX_NEW_TOKENS,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        try:
            return _parse_score(text)
        except (ValueError, TypeError) as e:
            print(f"[LLM] No score in the response: {e}")
            return {"implication_score": None, "reasoning": f"[parse error] {e}"}

    def _dispatch(self, prompt: str, _retries: int = 1) -> dict:
        """Send a fully-built prompt to Ollama and return the parsed JSON dict.

        A failed call (malformed JSON, missing score key, or transport error)
        is retried once before defaulting to 0.5, and every defaulted result
        carries a bracketed marker in `reasoning` so downstream analysis can
        count and exclude them.
        """
        if self.backend == "hf":
            return self._dispatch_hf(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                seed=42,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            if "implication_score" not in data:
                raise ValueError("response JSON lacks 'implication_score'")
            return {
                "implication_score": float(data["implication_score"]),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, ValueError) as e:
            if _retries > 0:
                print(f"[LLM] JSON Parsing Error (retrying): {e}")
                return self._dispatch(prompt, _retries - 1)
            print(f"[LLM] JSON Parsing Error: {e}")
            return {"implication_score": 0.5, "reasoning": f"[parse error] {e}"}
        except Exception as e:
            if _retries > 0:
                print(f"[LLM] Request failed (retrying, {type(e).__name__}): {e}")
                return self._dispatch(prompt, _retries - 1)
            print(f"[LLM] Request failed ({type(e).__name__}): {e}")
            return {"implication_score": 0.5, "reasoning": f"[request error] {e}"}

    def call_with_reasoning(self, premise: str, consequence: str) -> tuple:
        """Return (implication_score: float, reasoning: str)."""
        result = self._query(premise, consequence)
        return result["implication_score"], result["reasoning"]

    def __call__(self, premise: str, consequence: str) -> float:
        return self._query(premise, consequence)["implication_score"]