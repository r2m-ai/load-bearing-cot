"""Local vLLM backend with literal assistant-prefix continuation."""
from .protocol import REASONING, parse_reasoning_tags

MODEL_ALIASES = {
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}

# Reasoning delimiters by resolved model ID, for models whose chat template
# uses a pair other than the default. An entry is a convenience, not a
# validation of that model's reasoning format; --reasoning-tags overrides it.
MODEL_REASONING_TAGS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": REASONING,
}


def reasoning_tags_for(model_id, explicit=None):
    if explicit is not None:
        return parse_reasoning_tags(explicit)
    return MODEL_REASONING_TAGS.get(model_id, REASONING)


class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.model_id = MODEL_ALIASES.get(args.model, args.model)
        self.reasoning = reasoning_tags_for(self.model_id, getattr(args, "reasoning_tags", None))
        self.model = LLM(model=self.model_id, revision=args.model_revision,
                         tokenizer_revision=args.model_revision,
                         tensor_parallel_size=args.tensor_parallel_size,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         seed=args.seed)
        self.tokenizer = self.model.get_tokenizer()
        self.params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                                     seed=args.seed, skip_special_tokens=False)
        self.max_length = self.model.llm_engine.model_config.max_model_len

    def environment(self):
        """GPU, CUDA and dtype as the loaded process sees them; unknown values are None.

        Read here rather than in evaluate.py so the CLI stays free of GPU
        imports and tests keep running on the standard library.
        """
        import subprocess
        import torch

        def read(function):
            try:
                return function()
            except Exception:
                return None

        gpus = []
        for index in range(read(lambda: torch.cuda.device_count()) or 0):
            gpus.append({"name": read(lambda: torch.cuda.get_device_name(index)),
                         "compute_capability": read(lambda: ".".join(map(str, torch.cuda.get_device_capability(index))))})
        driver = read(lambda: subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10).split("\n")[0].strip() or None)
        return {"gpu": gpus, "device_count": len(gpus),
                "cuda": read(lambda: torch.version.cuda), "driver": driver,
                "torch": read(lambda: torch.__version__),
                "dtype": read(lambda: str(self.model.llm_engine.model_config.dtype)),
                "max_model_len": read(lambda: self.max_length)}

    def generate(self, requests):
        """requests: (user prompt, assistant prefix); no follow-up user turn."""
        from vllm.inputs import TokensPrompt
        open_tag, close_tag = self.reasoning
        prompts, open_thinking = [], []
        for prompt, prefix in requests:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
            ids = self.tokenizer.encode(formatted + prefix, add_special_tokens=False)
            if len(ids) + self.params.max_tokens > self.max_length:
                raise ValueError("Prefix plus generation budget exceeds model context; reduce --max-new-tokens")
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            open_thinking.append(not prefix and formatted.rfind(open_tag) > formatted.rfind(close_tag))
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
            # Some templates open the reasoning block themselves. An unfinished
            # generation must still be rejected by the eligibility gate.
            if opened and close_tag not in text:
                text = open_tag + text
            texts.append(text)
        return texts
