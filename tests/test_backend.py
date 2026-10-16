"""Validate backend wiring without loading GPU packages or model weights."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from load_bearing.backend import MODEL_REASONING_TAGS, VLLMBackend, reasoning_tags_for
from load_bearing.protocol import REASONING, reasoning_steps


class BackendTests(unittest.TestCase):
    def backend(self, template="USER:Q\nASSISTANT:", reasoning=REASONING):
        backend = VLLMBackend.__new__(VLLMBackend)
        backend.reasoning = reasoning
        backend.params = SimpleNamespace(max_tokens=10)
        backend.max_length = 1000
        class Tokenizer:
            eos_token = "<eos>"
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert messages == [{"role": "user", "content": "Q"}]
                assert not tokenize and add_generation_prompt
                return template
            def encode(self, text, add_special_tokens):
                assert not add_special_tokens
                return list(text.encode())
        backend.tokenizer = Tokenizer()
        backend.seen = []
        def generate(prompts, params):
            backend.seen.extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"Answer: {i}<eos>")])
                    for i in range(len(prompts))]
        backend.model = SimpleNamespace(generate=generate)
        return backend

    def setUp(self):
        module = ModuleType("vllm.inputs")
        module.TokensPrompt = dict
        self.modules = patch.dict(sys.modules, {"vllm.inputs": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_literal_prefix_and_order_above_ten_requests(self):
        backend = self.backend()
        texts = backend.generate([("Q", f"step {i}\n") for i in range(12)])
        self.assertEqual(texts, [f"Answer: {i}" for i in range(12)])
        self.assertEqual(bytes(backend.seen[0]["prompt_token_ids"]).decode(),
                         "USER:Q\nASSISTANT:step 0\n")

    def test_context_overflow_is_not_truncated(self):
        backend = self.backend()
        backend.max_length = 5
        with self.assertRaisesRegex(ValueError, "exceeds model context"):
            backend.generate([("Q", "prefix")])
        self.assertEqual(backend.seen, [])

    def test_template_opened_unfinished_thinking_is_ineligible(self):
        text = self.backend("USER:Q\nASSISTANT:<think>\n").generate([("Q", "")])[0]
        self.assertEqual(reasoning_steps(text)[1], [])

    def test_template_opened_custom_tags_are_closed_with_the_same_pair(self):
        tags = ("<reasoning>", "</reasoning>")
        text = self.backend("USER:Q\nASSISTANT:<reasoning>\n", tags).generate([("Q", "")])[0]
        self.assertTrue(text.startswith("<reasoning>"))
        self.assertEqual(reasoning_steps(text, tags)[1], [])
        # The default pair does not see a block at all, so the same text would
        # be read as plain visible steps: that is the silent path --reasoning-tags closes.
        self.assertEqual(reasoning_steps(text)[1], ["<reasoning>Answer: 0"])

    def test_reasoning_tags_resolution(self):
        self.assertEqual(reasoning_tags_for("unknown/model"), REASONING)
        self.assertEqual(reasoning_tags_for("unknown/model", "<r>,</r>"), ("<r>", "</r>"))
        for model_id, tags in MODEL_REASONING_TAGS.items():
            self.assertEqual(reasoning_tags_for(model_id), tags)
        for bad in ("<r>", "<r>,<r>", ",</r>"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reasoning_tags_for("unknown/model", bad)

    def test_environment_reads_what_torch_exposes_and_tolerates_gaps(self):
        torch = ModuleType("torch")
        torch.__version__ = "2.t"
        torch.version = SimpleNamespace(cuda="12.4")
        torch.cuda = SimpleNamespace(device_count=lambda: 2,
                                     get_device_name=lambda i: f"GPU{i}",
                                     get_device_capability=lambda i: (9, 0) if i == 0 else 1 / 0)
        backend = self.backend()
        backend.model = SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(dtype="torch.bfloat16")))
        with patch.dict(sys.modules, {"torch": torch}), \
             patch("subprocess.check_output", side_effect=FileNotFoundError):
            environment = backend.environment()
        self.assertEqual(environment["gpu"], [{"name": "GPU0", "compute_capability": "9.0"},
                                              {"name": "GPU1", "compute_capability": None}])
        self.assertEqual(environment["device_count"], 2)
        self.assertEqual(environment["cuda"], "12.4")
        self.assertIsNone(environment["driver"])
        self.assertEqual(environment["dtype"], "torch.bfloat16")
        self.assertEqual(environment["max_model_len"], 1000)
        torch.cuda = SimpleNamespace()
        with patch.dict(sys.modules, {"torch": torch}), \
             patch("subprocess.check_output", return_value="550.90\n"):
            environment = backend.environment()
        self.assertEqual(environment["gpu"], [])
        self.assertEqual(environment["driver"], "550.90")

    def test_empty_batch(self):
        self.assertEqual(self.backend().generate([]), [])
