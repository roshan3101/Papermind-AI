"""
Multi-provider LLM client.
Provider is selected exclusively via LLM_PROVIDER in .env.
Supported: openai | openrouter | gemini | anthropic
"""
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── System prompts ────────────────────────────────────────────────────────────

QA_SYSTEM = (
    "You are PaperMind AI, an expert research assistant. "
    "Answer questions based ONLY on the provided research paper excerpts. "
    "Always cite your sources by referencing [Paper: title, Page: N] for each claim. "
    "If the context doesn't contain sufficient information, say so clearly."
)

SUMMARY_SYSTEM = (
    "You are an expert academic summarizer. "
    "Provide structured summaries of research papers based on the given text. "
    "Be concise, accurate, and academic in tone."
)

COMPARE_SYSTEM = (
    "You are an expert research analyst specializing in comparing academic papers. "
    "Provide detailed, objective comparisons based only on the provided paper excerpts."
)


# ── Message builders ──────────────────────────────────────────────────────────

def build_qa_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[Source {i+1}: {c['paper_title']}, Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(context_chunks)
    )
    return [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def build_summary_messages(paper_text: str, title: str) -> list[dict]:
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": (
            f"Paper: {title}\n\nText:\n{paper_text[:12000]}\n\n"
            "Provide a structured summary with these sections:\n"
            "1. SUMMARY (2-3 sentences)\n"
            "2. KEY_CONTRIBUTIONS (bullet list)\n"
            "3. METHODOLOGY\n"
            "4. RESULTS\n"
            "5. LIMITATIONS"
        )},
    ]


def build_compare_messages(
    paper1_text: str, paper1_title: str,
    paper2_text: str, paper2_title: str,
) -> list[dict]:
    return [
        {"role": "system", "content": COMPARE_SYSTEM},
        {"role": "user", "content": (
            f"Paper 1: {paper1_title}\n{paper1_text[:6000]}\n\n"
            f"Paper 2: {paper2_title}\n{paper2_text[:6000]}\n\n"
            "Compare across:\n"
            "1. METHODOLOGY\n2. DATASETS\n3. PERFORMANCE_METRICS\n4. CONCLUSIONS\n5. OVERALL_COMPARISON"
        )},
    ]


# ── Provider implementations ──────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_openai(messages: list[dict], model: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or "", resp.usage.total_tokens if resp.usage else 0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_openrouter(messages: list[dict], model: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or "", resp.usage.total_tokens if resp.usage else 0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_gemini(messages: list[dict], model: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    # Build contents (non-system messages)
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=m["content"])]))

    config = gtypes.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction,
    )
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    text = response.text or ""
    tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
    return text, tokens


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_anthropic(messages: list[dict], model: str, temperature: float, max_tokens: int) -> tuple[str, int]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Extract system prompt; Anthropic takes it as a separate param
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=user_messages,
    )
    text = resp.content[0].text if resp.content else ""
    tokens = resp.usage.input_tokens + resp.usage.output_tokens if resp.usage else 0
    return text, tokens


# ── Public interface ──────────────────────────────────────────────────────────

async def chat_completion(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, int]:
    """Route to the correct provider based on LLM_PROVIDER env var."""
    provider = settings.LLM_PROVIDER
    logger.info("llm_call", provider=provider)

    if provider == "openai":
        return await _call_openai(messages, settings.OPENAI_MODEL, temperature, max_tokens)
    elif provider == "openrouter":
        return await _call_openrouter(messages, settings.OPENROUTER_MODEL, temperature, max_tokens)
    elif provider == "gemini":
        return await _call_gemini(messages, settings.GEMINI_MODEL, temperature, max_tokens)
    elif provider == "anthropic":
        return await _call_anthropic(messages, settings.ANTHROPIC_MODEL, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Must be one of: openai, openrouter, gemini, anthropic")
