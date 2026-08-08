import os
from typing import List, Optional, Any
from openai import OpenAI
from google import genai
from anthropic import Anthropic


class GPT5Client:
    """
    A framework for interacting with GPT-5 Large Language Model using OpenAI's official client.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        """
        Initialize the GPT-5 client.

        Args:
            api_key (str, optional): OpenAI API key. If None, will try to get from environment.
            model (str): Model name to use. Defaults to "gpt-5".
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from GPT-5.

        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            str: The generated response
        """
        try:
            # Note: GPT-5.4+ does not support temperature parameter
            create_kwargs = dict(
                model=self.model,
                reasoning={"effort": kwargs.get("reasoning_effort", "low")},
                input=prompt,
                max_output_tokens=kwargs.get("max_tokens", 10000),
            )
            # Only add temperature for models that support it (gpt-5 original)
            if 'gpt-5.' not in self.model:
                create_kwargs['temperature'] = kwargs.get("temperature", 0.0)
                create_kwargs['text'] = {"verbosity": "medium"}
            response = self.client.responses.create(**create_kwargs)
            return response.output_text

        except Exception as e:
            raise Exception(f"Error generating response: {e}")

    def stream_response(self, prompt: str, **kwargs):
        """
        Generate a streaming response from GPT-5.

        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters for the API call

        Yields:
            str: Chunks of the generated response
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1000),
                stream=True,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "stream"]}
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"Error in streaming response: {e}")

    def stream_chat_completion(self, messages: List[dict], **kwargs):
        """
        Generate a streaming chat completion with conversation history.

        Args:
            messages (List[dict]): List of message objects with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Yields:
            str: Chunks of the generated response
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1000),
                stream=True,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "stream"]}
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"Error in streaming chat completion: {e}")


class GeminiClient:
    """
    Client for Google Gemini models using the google-genai SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3-flash-preview"):
        self.model = model
        self.client = genai.Client(
            api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=kwargs.get("max_tokens", 65536),
                    temperature=kwargs.get("temperature", 0.0),
                ),
            )
            # Separate thinking from output
            parts = response.candidates[0].content.parts
            thinking_parts = []
            output_parts = []
            for p in parts:
                if getattr(p, 'thought', False):
                    thinking_parts.append(p.text)
                else:
                    output_parts.append(p.text)
            self._last_thinking = '\n'.join(thinking_parts) if thinking_parts else ''
            if output_parts:
                return '\n'.join(output_parts)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating response from Gemini: {e}")


class ClaudeClient:
    """
    Client for Anthropic Claude models.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-6"):
        self.model = model
        self.client = Anthropic(
            api_key=api_key or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            create_kwargs = dict(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 16000),
                messages=[{"role": "user", "content": prompt}],
            )
            # Enable extended thinking for reasoning-heavy tasks
            if kwargs.get("reasoning_effort"):
                effort = kwargs["reasoning_effort"]
                create_kwargs["thinking"] = {
                    "type": "adaptive",
                }
                # temperature must be 1 when thinking is enabled
                create_kwargs["temperature"] = 1
            else:
                create_kwargs["temperature"] = kwargs.get("temperature", 0.0)

            message = self.client.messages.create(**create_kwargs)

            # Extract only text blocks (skip thinking blocks)
            text_parts = [b.text for b in message.content if b.type == "text"]
            return '\n'.join(text_parts)
        except Exception as e:
            raise Exception(f"Error generating response from Claude ({self.model}): {e}")


class DeepSeekClient:
    """
    Client for DeepSeek's official API (OpenAI-compatible).
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DS_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def _is_valid_response(self, content: str) -> bool:
        """Check if response contains actual plan content."""
        if not content or not content.strip():
            return False
        # A valid plan should have at least one numbered action step
        import re
        return bool(re.search(r'\d+\.\s*\[', content))

    def generate_response(self, prompt: str, **kwargs) -> str:
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 8192),
                    **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
                )
                content = response.choices[0].message.content
                if self._is_valid_response(content):
                    return content
                print(f"    [DeepSeek] Empty/invalid response on attempt {attempt + 1}/{max_retries}, retrying...")
                time.sleep(2 ** attempt)  # exponential backoff
            except Exception as e:
                print(f"    [DeepSeek] Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"Error generating response from DeepSeek ({self.model}) after {max_retries} attempts: {e}")
        raise Exception(f"DeepSeek returned invalid response after {max_retries} attempts")


class VLLMClient:
    """
    Client for models served via vLLM's OpenAI-compatible API.
    Works with Llama and any other model hosted on vLLM.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "default",
                 base_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 10000),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response from vLLM ({self.model}): {e}")


class HuggingFaceClient:
    """
    Client for locally-hosted HuggingFace models via transformers.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Either a local checkpoint directory or a HuggingFace hub id.
        model = model or os.getenv("HF_MODEL_PATH", "meta-llama/Llama-3.3-70B-Instruct")
        self.model_path = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            temperature = kwargs.get("temperature", 0.0)
            max_tokens = kwargs.get("max_tokens", 10000)
            do_sample = temperature > 0

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            generated = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as e:
            raise Exception(f"Error generating response from HuggingFace local model ({self.model_path}): {e}")


class LLMClient:
    """
    Unified client that supports multiple LLM model families.
    Factory class for creating appropriate client based on model_family.
    """

    def __init__(self, model_family: str = "gpt", api_key: Optional[str] = None,
                 model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the unified LLM client.

        Args:
            model_family (str): The model family to use.
                Options: "gpt", "gemini", "llama", "deepseek"
            api_key (str, optional): API key for the selected model family
            model (str, optional): Specific model name. If None, uses default for the family.
            base_url (str, optional): vLLM server URL for llama/deepseek.

        Examples:
            client = LLMClient(model_family="gpt")
            client = LLMClient(model_family="gemini")
            client = LLMClient(model_family="llama", base_url="http://localhost:8000/v1")
            client = LLMClient(model_family="deepseek", base_url="http://localhost:8000/v1")
        """
        self.model_family = model_family.lower()

        if self.model_family == "gpt":
            self.client = GPT5Client(api_key=api_key, model=model or "gpt-5")
        elif self.model_family == "gemini":
            self.client = GeminiClient(api_key=api_key, model=model or "gemini-3-flash-preview")
        elif self.model_family == "llama":
            self.client = HuggingFaceClient(
                api_key=api_key,
                model=model or os.getenv("HF_MODEL_PATH",
                                         "meta-llama/Llama-3.3-70B-Instruct"),
            )
        elif self.model_family == "deepseek":
            self.client = DeepSeekClient(
                api_key=api_key,
                model=model or "deepseek-chat",
            )
        elif self.model_family == "claude":
            self.client = ClaudeClient(
                api_key=api_key,
                model=model or "claude-opus-4-6",
            )
        elif self.model_family == "qwen":
            self.client = VLLMClient(
                api_key=api_key,
                model=model or os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-27B-Instruct"),
                base_url=base_url or os.getenv("VLLM_URL", "http://localhost:8000/v1"),
            )
        else:
            raise ValueError(
                f"Unsupported model_family: {model_family}. "
                f"Supported: 'gpt', 'gemini', 'llama', 'deepseek', 'claude', 'qwen'"
            )

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the selected model.

        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters

        Returns:
            str: The generated response
        """
        return self.client.generate_response(prompt, **kwargs)



def main():
    """
    Example usage of the LLM clients.
    """
    try:
        print("=" * 60)
        print("Testing LLM Client with Multiple Model Families")
        print("=" * 60)

        prompt = "What is 2+2? Answer with just the number."

        # Example 1: GPT-5
        print("\n1. Testing GPT5Client:")
        unified_gpt = LLMClient(model_family="gpt")
        print(f"  Model family: {unified_gpt.model_family}")
        # response = unified_gpt.generate_response(prompt)
        # Example 2: Gemini
        print("\n2. Testing GeminiClient:")
        unified_gemini = LLMClient(model_family="gemini")
        print(f"  Model family: {unified_gemini.model_family}")

        # Example 3: Llama via vLLM
        print("\n3. Testing VLLMClient (Llama):")
        unified_llama = LLMClient(model_family="llama")
        print(f"  Model family: {unified_llama.model_family}")

        # Example 4: DeepSeek via vLLM
        print("\n4. Testing VLLMClient (DeepSeek):")
        unified_ds = LLMClient(model_family="deepseek")
        print(f"  Model family: {unified_ds.model_family}")

        print("\n" + "=" * 60)
        print("All clients initialized successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
