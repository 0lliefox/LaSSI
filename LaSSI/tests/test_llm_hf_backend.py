"""The LLM baselines' Hugging Face backend (evaluation plan, section 4.5, 19 Nov 2026).

No model is loaded or downloaded: the tokenizer and model are stand-ins, so these tests check the plumbing only
(backend choice, pinned revisions, greedy decoding, score parsing, and no default score on a failed response)."""
import json
import os
import unittest
from unittest import mock

import torch

from LaSSI.similarities import LLM


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __init__(self, reply):
        self.reply = reply

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        assert messages == [{"role": "user", "content": messages[0]["content"]}] and add_generation_prompt
        self.prompt = messages[0]["content"]
        return torch.tensor([[5, 6, 7]])

    def decode(self, ids, skip_special_tokens):
        assert ids.tolist() == [9, 9], ids
        return self.reply


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, ids, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([ids[0].tolist() + [9, 9]])


class FakeTokenizerV5(FakeTokenizer):
    """transformers 5's apply_chat_template(return_tensors="pt") gives a BatchEncoding, not the ids tensor (Comet, 21 Nov:
    gemma4:31b loaded on two L40S, then every prompt failed with "ones_like(): argument 'input' must be Tensor, not
    BatchEncoding")."""

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        from transformers import BatchEncoding
        ids = super().apply_chat_template(messages, add_generation_prompt, return_tensors)
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def install(model_name, reply, tokenizer=FakeTokenizer):
    tok, model = tokenizer(reply), FakeModel()
    LLM._HF_LOADED[model_name] = (tok, model, LLM.threading.Lock())
    return tok, model


class TestBackendChoice(unittest.TestCase):
    def tearDown(self):
        LLM._HF_LOADED.clear()

    def test_default_is_ollama_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LASSI_LLM_BACKEND", None)
            p = LLM.LLMPrompt("qwen2.5:7b")
        self.assertEqual(p.backend, "ollama")
        reply = mock.Mock()
        reply.choices = [mock.Mock(message=mock.Mock(content='{"reasoning": "r", "implication_score": 1.0}'))]
        p.client = mock.Mock()
        p.client.chat.completions.create.return_value = reply
        self.assertEqual(p("a", "b"), 1.0)
        kw = p.client.chat.completions.create.call_args.kwargs
        self.assertEqual((kw["temperature"], kw["seed"]), (0.1, 42))

    def test_unknown_backend_refused(self):
        with mock.patch.dict(os.environ, {"LASSI_LLM_BACKEND": "vllm"}):
            with self.assertRaises(ValueError):
                LLM.LLMPrompt("qwen2.5:7b")

    def test_hf_needs_a_models_file(self):
        with mock.patch.dict(os.environ, {"LASSI_LLM_BACKEND": "hf"}):
            os.environ.pop("LASSI_LLM_HF_MODELS", None)
            with self.assertRaises(RuntimeError):
                LLM.LLMPrompt("qwen2.5:7b")

    def test_hf_needs_a_pinned_revision(self):
        path = os.path.join(os.path.dirname(__file__), "_tmp_hf_models.json")
        with open(path, "w") as f:
            json.dump({"qwen2.5:7b": {"repo": "Qwen/Qwen2.5-7B-Instruct"}}, f)
        try:
            with mock.patch.dict(os.environ, {"LASSI_LLM_BACKEND": "hf", "LASSI_LLM_HF_MODELS": path}):
                with self.assertRaises(RuntimeError):
                    LLM.LLMPrompt("qwen2.5:7b")
        finally:
            os.remove(path)

    def test_hf_loads_from_the_local_cache_only(self):
        path = os.path.join(os.path.dirname(__file__), "_tmp_hf_models.json")
        with open(path, "w") as f:
            json.dump({"qwen2.5:7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "revision": "abc123"}}, f)
        try:
            with mock.patch.dict(os.environ, {"LASSI_LLM_HF_MODELS": path}), \
                    mock.patch("transformers.AutoTokenizer.from_pretrained") as tok, \
                    mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as mod:
                LLM._hf_load("qwen2.5:7b")
            for call in (tok.call_args, mod.call_args):
                self.assertEqual(call.args, ("Qwen/Qwen2.5-7B-Instruct",))
                self.assertEqual((call.kwargs["revision"], call.kwargs["local_files_only"]), ("abc123", True))
            self.assertEqual(mod.call_args.kwargs["torch_dtype"], "auto")
        finally:
            os.remove(path)


class TestHFDispatch(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"LASSI_LLM_BACKEND": "hf"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        LLM._HF_LOADED.clear()

    def test_greedy_with_the_published_prompt(self):
        tok, model = install("qwen2.5:7b", '{"reasoning": "same place", "implication_score": 0.5}')
        p = LLM.LLMPrompt("qwen2.5:7b")
        self.assertEqual(p.call_with_reasoning("E", "C"), (0.5, "same place"))
        self.assertEqual(tok.prompt, LLM.LLMPrompt._PROMPT_TEMPLATE.format(premise="E", consequence="C"))
        kw = model.kwargs
        self.assertIs(kw["do_sample"], False)
        self.assertIsNone(kw["temperature"])
        self.assertEqual(kw["max_new_tokens"], 512)
        self.assertEqual(kw["pad_token_id"], 2)

    def test_transformers5_chat_template_output(self):
        tok, model = install("gemma4:31b", '{"reasoning": "closed", "implication_score": 1.0}', FakeTokenizerV5)
        self.assertEqual(LLM.LLMPrompt("gemma4:31b").call_with_reasoning("E", "C"), (1.0, "closed"))
        self.assertEqual(model.kwargs["attention_mask"].tolist(), [[1, 1, 1]])

    def test_a_generation_fault_is_raised_not_scored_none(self):
        tok, model = install("gemma4:31b", "unused")
        model.generate = mock.Mock(side_effect=TypeError("a fault in the set-up"))
        with self.assertRaises(TypeError):
            LLM.LLMPrompt("gemma4:31b").call_with_reasoning("E", "C")

    def test_json_inside_text(self):
        install("llama3.2:3b", 'Here you go:\n```json\n{"reasoning": "no", "implication_score": 0.0}\n```')
        self.assertEqual(LLM.LLMPrompt("llama3.2:3b")("E", "C"), 0.0)

    def test_no_score_gives_none_not_a_default(self):
        install("llama3.2:3b", "I cannot decide.")
        score, reason = LLM.LLMPrompt("llama3.2:3b").call_with_reasoning("E", "C")
        self.assertIsNone(score)
        self.assertTrue(reason.startswith("[parse error]"))

    def test_non_numeric_score_gives_none(self):
        install("llama3.2:3b", '{"reasoning": "x", "implication_score": "maybe"}')
        self.assertIsNone(LLM.LLMPrompt("llama3.2:3b")("E", "C"))


if __name__ == "__main__":
    unittest.main()
