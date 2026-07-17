"""Model release-date table for E4.

Policy:
- Dated model strings: use the embedded snapshot date as the release date
  (snapshot dates trail public availability by <= 8 days for the models here;
  negligible on a 17-month axis).
- Undated alias strings: mapped to the snapshot they resolved to during the
  observed run window, with a web-verified public release date and source.
- Excluded: `openai/ft:gpt-4o-*` fine-tunes (uk-dsit; not public model
  releases), `vllm/*` locally hosted checkpoints (the base models have public
  release dates — e.g. Llama-3.1-8B-Instruct 2024-07-23 — but the served
  weights' provenance/modification status is unverifiable, so they are not
  treated as observations of the public release; 5 rows total, immaterial),
  and `mistralazure/Mistral-Large-2411-CAST` (modified deployment variant of
  Mistral Large 2411, not a distinct public release).
"""

# canonical name -> (release_date ISO, source)
RELEASE_DATES: dict[str, tuple[str, str]] = {
    "claude-3-haiku": (
        "2024-03-07",
        "snapshot date embedded in model string claude-3-haiku-20240307",
    ),
    "claude-3-5-sonnet-20240620": (
        "2024-06-20",
        "snapshot date embedded in model string",
    ),
    "gpt-4o-2024-08-06": (
        "2024-08-06",
        "snapshot date embedded in model string; `openai/gpt-4o` alias resolved "
        "to this snapshot in the run window (Jan 2025)",
    ),
    "o1-mini": (
        "2024-09-12",
        "OpenAI o1-preview/o1-mini launch 2024-09-12 "
        "(openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning)",
    ),
    "claude-3-5-sonnet-20241022": (
        "2024-10-22",
        "Anthropic upgraded Claude 3.5 Sonnet release 2024-10-22 "
        "(platform.claude.com release notes); `-latest` alias pointed at "
        "20241022 during the run window (2025-01..2025-06)",
    ),
    "mistral-large-2411": (
        "2024-11-19",
        "Mistral Large 24.11 released 2024-11-19 (docs.mistral.ai; "
        "huggingface.co/mistralai/Mistral-Large-Instruct-2411)",
    ),
    "o1-2024-12-17": (
        "2024-12-17",
        "snapshot date embedded in model string (o1 API release); "
        "`openai/o1` alias has only this API snapshot",
    ),
    "o3-mini": (
        "2025-01-31",
        "OpenAI o3-mini launch 2025-01-31 (openai.com/index/openai-o3-mini; "
        "TechCrunch 2025-01-31)",
    ),
    "claude-3-7-sonnet": (
        "2025-02-19",
        "snapshot date embedded in claude-3-7-sonnet-20250219; `-latest` alias "
        "has only this snapshot",
    ),
    "gemma-3-27b-it": (
        "2025-03-12",
        "Google 'Introducing Gemma 3' blog 2025-03-12 "
        "(ai.google.dev/gemma/docs/releases)",
    ),
    "o3-2025-04-16": (
        "2025-04-16",
        "OpenAI o3/o4-mini launch 2025-04-16 (community.openai.com "
        "announcement t/1230164; TechCrunch 2025-04-16); `openai/o3` alias "
        "has only this snapshot",
    ),
    "o4-mini-2025-04-16": (
        "2025-04-16",
        "OpenAI o3/o4-mini launch 2025-04-16 (same sources); `openai/o4-mini` "
        "alias has only this snapshot",
    ),
    "claude-sonnet-4": (
        "2025-05-14",
        "snapshot date embedded in claude-sonnet-4-20250514 (public launch "
        "2025-05-22)",
    ),
    "claude-opus-4": (
        "2025-05-14",
        "snapshot date embedded in claude-opus-4-20250514 (public launch "
        "2025-05-22)",
    ),
    "claude-opus-4-1": (
        "2025-08-05",
        "snapshot date embedded in claude-opus-4-1-20250805",
    ),
    "gpt-5": (
        "2025-08-07",
        "snapshot date embedded in gpt-5-2025-08-07; GPT-5 launch 2025-08-07 "
        "(techcrunch.com/2025/08/07/openais-gpt-5-is-here); `openai/gpt-5` "
        "alias resolved to this snapshot in the run window",
    ),
}

# raw model string -> canonical name (None = excluded, with reason)
CANONICAL: dict[str, str] = {
    "anthropic/claude-3-haiku-20240307": "claude-3-haiku",
    "anthropic/claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "anthropic/claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "anthropic/claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    "anthropic/claude-3-7-sonnet-latest": "claude-3-7-sonnet",
    "anthropic/claude-sonnet-4-20250514": "claude-sonnet-4",
    "anthropic/claude-opus-4-20250514": "claude-opus-4",
    "anthropic/claude-opus-4-1-20250805": "claude-opus-4-1",
    "openai/gpt-4o": "gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "openai/o1": "o1-2024-12-17",
    "openai/o1-2024-12-17": "o1-2024-12-17",
    "openai/o1-mini": "o1-mini",
    "openai/o3": "o3-2025-04-16",
    "openai/o3-mini": "o3-mini",
    "openai/o4-mini": "o4-mini-2025-04-16",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-5-2025-08-07": "gpt-5",
    "azureai/Mistral-Large-2411": "mistral-large-2411",
    "mistralazure/Mistral-Large-2411": "mistral-large-2411",
    "gemma/gemma-3-27b-it": "gemma-3-27b-it",
}


def exclusion_reason(raw_model: str) -> str | None:
    """Reason a raw model string is excluded, or None if it is mappable."""
    if raw_model in CANONICAL:
        return None
    if raw_model.startswith("openai/ft:gpt-4o"):
        return "uk-dsit fine-tune of gpt-4o (not a public model release)"
    if raw_model.startswith("vllm/"):
        return (
            "locally hosted vllm checkpoint (base model has a public release "
            "date, but served-weight provenance is unverifiable)"
        )
    if "CAST" in raw_model:
        return "modified deployment variant (Mistral-Large-2411-CAST)"
    return "unrecognised model string (cannot be dated)"
