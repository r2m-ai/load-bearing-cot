"""Local vLLM backend with literal assistant-prefix continuation."""

MODEL_ALIASES = {
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}


class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.model_id = MODEL_ALIASES.get(args.model, args.model)
        self.model = LLM(model=self.model_id, revision=args.model_revision,
                         tokenizer_revision=args.model_revision,
                         tensor_parallel_size=args.tensor_parallel_size,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         seed=args.seed)
        self.tokenizer = self.model.get_tokenizer()
        self.params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                                     seed=args.seed, skip_special_tokens=False)
        self.max_length = self.model.llm_engine.model_config.max_model_len

    def generate(self, requests):
        """requests: (user prompt, assistant prefix); no follow-up user turn."""
        from vllm.inputs import TokensPrompt
        prompts, open_thinking = [], []
        for prompt, prefix in requests:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
            ids = self.tokenizer.encode(formatted + prefix, add_special_tokens=False)
            if len(ids) + self.params.max_tokens > self.max_length:
                raise ValueError("Prefix plus generation budget exceeds model context; reduce --max-new-tokens")
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            open_thinking.append(not prefix and formatted.rfind("<think>") > formatted.rfind("</think>"))
        if not prompts:
            return []
        outputs = self.model.generate(prompts, self.params)
        if len(outputs) != len(prompts):
            raise ValueError("Backend returned the wrong number of generations")
        # vLLM returns outputs in input order. Sorting string request IDs breaks
        # correspondence after request 9.
        texts = []
        for output, opened in zip(outputs, open_thinking):
            text = output.outputs[0].text
            # Preserve reasoning tags, remove only a terminal EOS marker.
            for token in (self.tokenizer.eos_token, "<|eot_id|>", "<end_of_turn>"):
                if token and text.endswith(token):
                    text = text[:-len(token)]
            # Some templates open <think> themselves. An unfinished generation
            # must still be rejected by the visible-rationale eligibility gate.
            if opened and "</think>" not in text:
                text = "<think>" + text
            texts.append(text)
        return texts
