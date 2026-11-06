"""
Day 1 — LLM API Foundation (VinUni Codelab Day 01)

Chạy từ thư mục gốc project:
    python starter-code/template.py
Chạy test:
    pytest tests/ -v

API key được đọc từ file .env ở thư mục gốc (KHÔNG commit file .env):
    OPENAI_API_KEY=sk-proj-...
    GEMINI_API_KEY=AIzaSy...
    # ANTHROPIC_API_KEY=sk-ant-...     (tuỳ chọn)
    # GEMINI_MODEL=gemini-2.5-flash    (tuỳ chọn, đổi model Gemini)
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp .env ở thư mục gốc project, dù chạy script từ thư mục nào.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------------------
# Bảng giá USD / 1 triệu token (theo đề bài; test chấm theo bảng này)
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
# Model theo đề bài (pytest chấm chi phí theo model này).
GEMINI_MODEL = "gemini-2.5-flash"
# Khi chạy thật: gemini-2.5-flash đã ngừng cấp cho tài khoản mới (404),
# nên dùng model hiện hành. Có thể đổi bằng GEMINI_MODEL=... trong .env.
LIVE_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# claude-3-5-haiku đã ngừng hoạt động -> dùng model Haiku hiện hành.
ANTHROPIC_MODEL = "claude-haiku-4-5"


def calculate_cost(model: str, usage: dict) -> float:
    """Cost = (in * giá_in + out * giá_out) / 1,000,000"""
    prices = PRICING_1M_TOKENS.get(model)
    if prices is None:
        return 0.0
    return (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("output_tokens", 0) * prices["output"]
    ) / 1_000_000


def _gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _thinking_config(types, model: str) -> dict:
    """
    Gemini 2.5+ "suy nghĩ" ngầm và tiêu tốn max_output_tokens, làm câu trả
    lời bị cụt. Giảm mức suy nghĩ để token dành cho câu trả lời.
    """
    try:
        if "2.5" in model:
            return {"thinking_config": types.ThinkingConfig(thinking_budget=0)}
        return {"thinking_config": types.ThinkingConfig(thinking_level="low")}
    except Exception:  # SDK cũ chưa hỗ trợ tham số này
        return {}


API_TIMEOUT_SECONDS = 60  # quá thời gian này thì báo lỗi thay vì treo


def _gemini_http_options(types):
    """Giới hạn thời gian chờ & tắt retry ngầm của SDK (ta tự retry)."""
    try:
        return types.HttpOptions(
            timeout=API_TIMEOUT_SECONDS * 1000,  # đơn vị: mili giây
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except Exception:  # SDK cũ chưa có retry_options
        return types.HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000)


def _is_transient(exc: Exception) -> bool:
    """Chỉ lỗi tạm thời (quá tải/rate limit) mới đáng retry; 404, hết credit thì không."""
    msg = str(exc)
    if "credit" in msg.lower() or "quota" in msg.lower():
        return False
    return any(c in msg for c in ("503", "UNAVAILABLE", "429", "500", "RESOURCE_EXHAUSTED"))


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini (SDK mới google-genai)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_gemini_key(), http_options=_gemini_http_options(types))

    config_kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    # Gemini 2.5 "suy nghĩ" ngầm và ăn mất max_output_tokens -> câu trả lời
    # bị cụt. Tắt thinking để toàn bộ 256 token dành cho câu trả lời.
    thinking = _thinking_config(types, model)

    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs, **thinking),
        )
    except Exception as exc:
        # Model không nhận cấu hình thinking -> gọi lại không kèm nó.
        if not thinking or "thinking" not in str(exc).lower():
            raise
        start = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    latency = time.perf_counter() - start

    meta = response.usage_metadata
    usage = {
        "input_tokens": (meta.prompt_token_count or 0) if meta else 0,
        "output_tokens": (meta.candidates_token_count or 0) if meta else 0,
    }
    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (tuỳ chọn)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=API_TIMEOUT_SECONDS
    )

    # Model Claude mới không cho truyền cùng lúc temperature và top_p,
    # nên chỉ truyền temperature (top_p giữ trong chữ ký hàm theo đề).
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start

    text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models
# ---------------------------------------------------------------------------
def _run_one(call_fn: Callable, prompt: str, model: str, retry: bool) -> dict:
    """Gọi 1 model; nếu lỗi thì ghi lại lỗi, không làm hỏng model khác."""
    try:
        print(f"  -> Đang gọi {model} ...", flush=True)
        attempt = lambda: call_fn(prompt, model=model)  # noqa: E731
        try:
            text, latency, usage = attempt()
        except Exception as exc:
            if not (retry and _is_transient(exc)):
                raise
            text, latency, usage = retry_with_backoff(attempt, max_retries=3, base_delay=2)
        return {
            "model": model,
            "response": text,
            "latency": latency,
            "cost": calculate_cost(model, usage),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model,
            "response": "",
            "latency": None,
            "cost": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(exc),
        }


def compare_models(prompt: str) -> dict:
    return {
        # OpenAI hết credit (429) thì retry cũng vô ích -> không retry.
        "gpt4o": _run_one(call_openai, prompt, OPENAI_MODEL, retry=False),
        "gpt4o_mini": _run_one(call_openai, prompt, OPENAI_MINI_MODEL, retry=False),
        # Gemini hay bị 503 quá tải tạm thời -> retry với backoff.
        "gemini_flash": _run_one(call_gemini, prompt, GEMINI_MODEL, retry=True),
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot với Gemini (giữ 3 lượt gần nhất)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    if not _gemini_key():
        print("[Warning] Chưa có GEMINI_API_KEY trong .env -> không chạy chatbot.")
        return

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_gemini_key(), http_options=_gemini_http_options(types))
    config = types.GenerateContentConfig(
        temperature=0.7, top_p=0.9, max_output_tokens=1024,
        **_thinking_config(types, GEMINI_MODEL),
    )

    print("\n" + "=" * 50)
    print(f"  Gemini Chatbot ({GEMINI_MODEL}) — gõ 'quit'/'exit' để thoát")
    print("=" * 50 + "\n")

    history: list = []  # mỗi phần tử là types.Content (role user/model)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Tạm biệt!")
            break

        user_msg = types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        contents = history + [user_msg]

        print("Gemini: ", end="", flush=True)
        reply = ""
        try:
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=contents, config=config
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    reply += chunk.text
            print("\n")
        except Exception as exc:
            print(f"\n[Gemini Error] {exc}\nThử gửi lại sau vài giây.\n")
            continue

        history += [
            user_msg,
            types.Content(role="model", parts=[types.Part.from_text(text=reply)]),
        ]
        history = history[-6:]  # 3 lượt = 6 message


# ---------------------------------------------------------------------------
# Bonus A — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[Retry {attempt + 1}/{max_retries}] {type(exc).__name__}: "
                  f"thử lại sau {delay:.1f}s")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus C — Bảng Markdown
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "gpt4o": "GPT-4o",
    "gpt4o_mini": "GPT-4o-mini",
    "gemini_flash": "Gemini-Flash",
}


def format_comparison_table(results: list[dict]) -> str:
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        prompt = result.get("prompt", "")
        for key, label in MODEL_LABELS.items():
            stats = result.get(key)
            if not stats:
                continue
            if stats.get("error"):
                lines.append(f"| {prompt} | {label} | ERROR: {stats['error'][:50]} | - | - | - |")
                continue
            resp = (stats.get("response") or "").replace("\n", " ").replace("|", "/")
            if len(resp) > 50:
                resp = resp[:50] + "..."
            lines.append(
                f"| {prompt} | {label} | {resp} | {stats['latency']:.2f}s | "
                f"{stats['input_tokens']}/{stats['output_tokens']} | ${stats['cost']:.8f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chạy thử thủ công
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."

    # Chạy thật thì dùng model Gemini còn hoạt động.
    GEMINI_MODEL = LIVE_GEMINI_MODEL

    print("=== 1-2. So sánh GPT-4o / GPT-4o-mini / Gemini ===")
    result = compare_models(prompt)
    for name, stats in result.items():
        print(f"\n[{name.upper()}] ({stats['model']})")
        if stats["error"]:
            print(f"  Lỗi: {stats['error'][:200]}")
        else:
            print(f"  Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.8f} | "
                  f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"  Response: {stats['response']}")

    print("\n=== 3. Claude (tuỳ chọn) ===")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            text, lat, usage = call_anthropic(prompt)
            print(f"  {text}\n  Latency: {lat:.2f}s | Usage: {usage}")
        except Exception as exc:
            print(f"  Lỗi: {exc}")
    else:
        print("  Bỏ qua: chưa có ANTHROPIC_API_KEY trong .env")

    print("\n=== Bonus B/C: Bảng so sánh ===")
    result["prompt"] = prompt
    print(format_comparison_table([result]))

    print("\n=== 5. Streaming Chatbot ===")
    streaming_chatbot()
