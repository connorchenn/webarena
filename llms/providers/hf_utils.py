import openai
from text_generation import Client


def generate_from_huggingface_completion(
    prompt: str,
    model: str,
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    # Check if it's a vllm/OpenAI-compatible endpoint
    # Endpoints ending in /v1 or containing /v1/completions or /v1/chat/completions
    if (
        model_endpoint.endswith("/v1")
        or "/v1/completions" in model_endpoint
        or "/v1/chat/completions" in model_endpoint
    ):
        return generate_from_vllm_completion(
            prompt=prompt,
            model=model,
            model_endpoint=model_endpoint,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
    else:
        # Original text-generation-inference client
        client = Client(model_endpoint, timeout=60)
        generation: str = client.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        ).generated_text
        return generation


def generate_from_vllm_completion(
    prompt: str,
    model: str,
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    """Generate from vllm using OpenAI client (like openevolve does)."""
    base_url = model_endpoint

    client = openai.OpenAI(
        api_key="EMPTY",  # vllm doesn't require a real API key
        base_url=base_url,
        timeout=300.0,
        max_retries=3,
    )

    try:
        # Use chat completions endpoint (works better with vllm)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_sequences,
        )

        # Extract the generated text
        generation: str = response.choices[0].message.content
        return generation
    except Exception as e:
        raise RuntimeError(f"Error calling vllm endpoint: {e}")
