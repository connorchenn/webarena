from pathlib import Path
from typing import Any

import tiktoken
from jinja2 import Template
from transformers import AutoTokenizer, LlamaTokenizer


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name

        if provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            # Check if it's a Qwen model
            if (
                "Qwen" in model_name
                or "qwen" in model_name
                or "skyagent" in model_name.lower()
            ):
                # Use AutoTokenizer for Qwen models
                if "skyagent" in model_name.lower():
                    model_name = "Qwen/Qwen3-32B"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                # Configure tokenizer to preserve thinking tokens
                # Don't add special tokens automatically
                if hasattr(self.tokenizer, "add_special_tokens"):
                    self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
                if hasattr(self.tokenizer, "add_bos_token"):
                    self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
                if hasattr(self.tokenizer, "add_eos_token"):
                    self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]

                # Load custom chat template for Qwen models
                # This sets the chat template for client-side operations (token counting, etc.)
                # Note: For vllm deployments, also configure the server with the same template using:
                #   --chat-template /path/to/qwen3_acc_thinking.jinja2
                template_path = (
                    Path(__file__).parent.parent
                    / "agent"
                    / "prompts"
                    / "chat_templates"
                    / "qwen3_acc_thinking.jinja2"
                )
                if template_path.exists():
                    with open(template_path, "r") as f:
                        custom_template = f.read()
                    self.tokenizer.chat_template = custom_template
            else:
                # Use LlamaTokenizer for other models (e.g., Llama-2)
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
                # turn off adding special tokens automatically
                self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
                self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
                self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)
