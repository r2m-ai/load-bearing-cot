"""Validate backend wiring without loading GPU packages or model weights."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from load_bearing.backend import VLLMBackend
from load_bearing.protocol import reasoning_steps


class BackendTests(unittest.TestCase):
    def backend(self, template="USER:Q\nASSISTANT:"):
        backend = VLLMBackend.__new__(VLLMBackend)
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

    def test_empty_batch(self):
        self.assertEqual(self.backend().generate([]), [])
