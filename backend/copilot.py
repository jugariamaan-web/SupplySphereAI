"""
copilot.py
------------------------------------------------------------
AI Procurement Copilot - powered by Groq (official Python SDK).

Instead of fine-tuning our own LLM (way too complex for a
diploma project), we send Groq a short SYSTEM PROMPT containing
a summary of our real data (forecast, risk, inventory numbers)
plus the user's question. The LLM then answers in natural
language, grounded in our actual numbers.

This also powers the AI Report Generator: we simply ask the
LLM to turn the same data summary into a written report.

IMPORTANT - what Groq is (and is NOT) used for:
    USED for   : AI Copilot chat, AI Reports, executive summaries,
                 natural-language explanations, business insights.
    NEVER used for : demand forecasting (Random Forest / LSTM handle
                 that) or inventory-risk prediction (XGBoost handles
                 that) or any EOQ/safety-stock/ROP math (inventory.py
                 handles that with plain formulas). Groq only explains
                 numbers that were already computed elsewhere - it
                 never generates them.

Setup:
    backend/.env must contain:
        GROQ_API_KEY=your_api_key_here
    Replace the placeholder with a real key from https://console.groq.com/keys
    No code changes are needed after that - this module re-reads
    the key from the environment on every request, so simply saving
    the .env file (and restarting the server) activates the AI
    features immediately. Until then, Copilot and Reports both fall
    back to a professional, data-grounded response automatically.
------------------------------------------------------------
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("supplysphere")

# Resolve backend/.env explicitly (rather than relying on find_dotenv's
# cwd/frame-walking heuristics) so the key loads correctly no matter
# where the process was launched from (run_project.bat, an IDE "Run"
# button from the project root, Docker, etc).
ENV_PATH = Path(__file__).resolve().parent / ".env"

# The literal placeholder shipped in .env.example / .env - if this is
# still present (or the variable is missing/blank), we treat Groq as
# "not configured" rather than attempting a doomed API call.
PLACEHOLDER_KEY = "your_api_key_here"


def _load_env_fresh() -> None:
    """
    Re-reads backend/.env on every call instead of once at import time.

    Root cause of the original bug: GROQ_API_KEY was read into a
    module-level constant when copilot.py was first imported, and
    `load_dotenv()` (no args, no override) only searched from the
    caller's cwd/frame and never overwrote a variable that was
    already present in the process environment. In practice that
    meant:
      1) Editing backend/.env after the server had already started
         had no effect - uvicorn's --reload watcher only reacts to
         .py file changes, not .env, so the OLD key stayed cached in
         memory even after a "save + it should just work" edit.
      2) If GROQ_API_KEY was ever set at the OS/Docker level (e.g.
         docker-compose's `GROQ_API_KEY: ${GROQ_API_KEY:-your_api_key_here}`),
         that value would win over backend/.env forever, since
         load_dotenv() does not override existing env vars by default.
    Reading the key fresh from disk (with override=True) on every
    request fixes both cases: replacing the placeholder and
    restarting (or even just re-saving) the backend now reliably
    activates Groq with no code changes required.
    """
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def _get_groq_key() -> str:
    _load_env_fresh()
    return os.getenv("GROQ_API_KEY", "").strip()


# ROOT CAUSE (found via root-cause trace, see chat/PR history):
# The old default here was "llama-3.3-70b-versatile". Groq announced its
# deprecation on 2026-06-17 with a shutdown date of 2026-08-16
# (see https://console.groq.com/docs/deprecations). During this transition
# window deprecated models are deprioritized on Groq's infra and can return
# capacity/routing errors well before the hard shutdown date, which is
# exactly the "Groq request failed" pattern that lands users in
# _fallback_reply() even with a perfectly valid API key. Groq's own
# recommended replacement is "openai/gpt-oss-120b", which is what we
# default to now. GROQ_MODEL in .env still overrides this if set.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Model IDs that are fully decommissioned or in Groq's deprecation window -
# used only to produce a precise, actionable log message (and to trigger a
# single automatic retry on the safe default) instead of a generic failure.
_KNOWN_STALE_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}


def _get_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()


def _is_configured() -> bool:
    """True only if a real (non-empty, non-placeholder) key is present."""
    key = _get_groq_key()
    return bool(key) and key != PLACEHOLDER_KEY


def _mask(key: str) -> str:
    """gsk_abCD...wxyz - never log the full key."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:5]}...{key[-4:]} (len={len(key)})"


def _get_client():
    """
    Lazily creates the Groq client. Imported here (not at module load)
    so the whole backend still starts up cleanly even if the `groq`
    package were ever missing - it's only required at call time.
    Returns None if the key isn't configured or the client can't be built.
    """
    key = _get_groq_key()
    logger.info(f"[copilot.py] STAGE 1/2 - .env loaded from {ENV_PATH} (exists={ENV_PATH.exists()})")
    logger.info(f"[copilot.py] STAGE 2/2 - GROQ_API_KEY resolved: {_mask(key)}")

    if not key or key == PLACEHOLDER_KEY:
        logger.warning(
            "[copilot.py] GROQ_API_KEY is missing or still the placeholder value - "
            "Copilot will use the fallback reply. Set a real key in backend/.env."
        )
        return None

    try:
        import groq as groq_pkg
        from groq import Groq
        logger.info(f"[copilot.py] STAGE 4 - groq SDK imported OK (version={getattr(groq_pkg, '__version__', 'unknown')})")
    except Exception as e:
        logger.error(f"[copilot.py] STAGE 4 FAILED - could not import 'groq' package: {e}", exc_info=True)
        return None

    try:
        client = Groq(api_key=key)
        logger.info("[copilot.py] STAGE 5 - Groq client created OK")
        return client
    except Exception as e:
        logger.error(f"[copilot.py] STAGE 5 FAILED - could not create Groq client: {e}", exc_info=True)
        return None


def _call_groq(client, model: str, system_prompt: str, user_message: str):
    """Single request attempt. Raises on failure - callers decide what to do."""
    logger.info(f"[copilot.py] STAGE 6 - sending request | model={model} | prompt_chars={len(user_message)}")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,  # a bit factual, not too creative
        max_tokens=1024,
        timeout=30,
    )
    logger.info(f"[copilot.py] STAGE 7 - response received | id={getattr(completion, 'id', '?')} | "
                f"finish_reason={completion.choices[0].finish_reason if completion.choices else '?'}")
    return completion


def ask_copilot(user_message: str, context_data: dict, system_prompt: str = None) -> str:
    """
    Sends the user's question + a data summary to Groq and returns
    the AI's text reply. Never raises - any failure (missing key,
    network error, malformed response, rate limit, etc.) falls back
    to a professional, data-grounded reply instead.

    `context_data` is a plain dict built by main.py, e.g.:
        {
            "forecast_summary": "...",
            "risk_summary": "...",
            "inventory_summary": "...",
        }

    `system_prompt` is optional: pass a custom persona/format prompt
    (e.g. from generate_report()) to override the default chat-Copilot
    system prompt built by build_system_prompt(). This keeps the
    4-section chat format (Summary/Key Findings/Recommendation/
    Business Impact) separate from the 6-section report format, since
    both use this same Groq call path.
    """
    logger.info("[copilot.py] STAGE 3 - ask_copilot() called")
    client = _get_client()
    if client is None:
        return _fallback_reply(context_data)

    if system_prompt is None:
        system_prompt = build_system_prompt(context_data)
    model = _get_model_name()

    try:
        completion = _call_groq(client, model, system_prompt, user_message)
        reply = completion.choices[0].message.content
        if reply:
            logger.info(f"[copilot.py] STAGE 8 - SUCCESS, reply_chars={len(reply)}")
            return reply
        logger.warning("[copilot.py] STAGE 8 - Groq returned an empty message content; using fallback.")
        return _fallback_reply(context_data)

    except Exception as e:
        logger.error(f"[copilot.py] STAGE 6/7 FAILED - Groq request failed with model='{model}': {e}", exc_info=True)

        # If the configured model is a known-stale/deprecated ID, that's very
        # likely the actual cause (see DEFAULT_MODEL comment above) - retry
        # once against DEFAULT_MODEL so a stale GROQ_MODEL in .env doesn't
        # permanently stick the user with the fallback reply.
        is_stale_model_error = (
            model in _KNOWN_STALE_MODELS
            or "decommissioned" in str(e).lower()
            or "model_decommissioned" in str(e).lower()
        )
        if is_stale_model_error and model != DEFAULT_MODEL:
            logger.warning(
                f"[copilot.py] '{model}' looks deprecated/decommissioned on Groq "
                f"(see https://console.groq.com/docs/deprecations). "
                f"Retrying once with the current default '{DEFAULT_MODEL}' - "
                f"update GROQ_MODEL in backend/.env to stop seeing this."
            )
            try:
                completion = _call_groq(client, DEFAULT_MODEL, system_prompt, user_message)
                reply = completion.choices[0].message.content
                if reply:
                    logger.info(f"[copilot.py] STAGE 8 - SUCCESS on retry with '{DEFAULT_MODEL}', reply_chars={len(reply)}")
                    return reply
            except Exception as e2:
                logger.error(f"[copilot.py] Retry with '{DEFAULT_MODEL}' also failed: {e2}", exc_info=True)

        # Covers Groq SDK errors (auth, rate limit, connection, timeout)
        # and any unexpected response shape. Never leak the raw
        # exception to the user - log it server-side and degrade
        # gracefully instead.
        return _fallback_reply(context_data)


def _fallback_reply(context_data: dict) -> str:
    """Professional message shown whenever Groq isn't configured or
    is unavailable. Still useful: it surfaces the real numbers we
    already have from the database and live calculations."""
    return (
        "AI Copilot is not configured yet, so here's a summary based on current system data:\n\n"
        f"- {context_data.get('forecast_summary', 'No forecast data available.')}\n"
        f"- {context_data.get('risk_summary', 'No risk data available.')}\n"
        f"- {context_data.get('inventory_summary', 'No inventory data available.')}\n\n"
        "To enable AI-powered answers, add a Groq API key to backend/.env "
        "(replace GROQ_API_KEY=your_api_key_here with a real key from "
        "console.groq.com) and restart the server."
    )


def build_system_prompt(context_data: dict) -> str:
    """
    Builds the system prompt sent to Groq for every Copilot turn.

    Persona: SupplySphere AI behaves as an Enterprise Inventory
    Consultant - concise, business-oriented, and executive-friendly -
    rather than a documentation bot. Every reply is grounded in the
    real numbers injected below (no hallucinated products/stores/
    figures), but the OUTPUT FORMAT and TONE are consultant-style,
    not academic. See the format contract and banned-phrasing list
    below for the specific rules this enforces.
    """
    return f"""You are the SupplySphere AI Enterprise Inventory Consultant - a senior supply chain advisor speaking directly to executives and operations leaders. You are not a technical assistant and you are not documentation; you give the kind of tight, confident briefing a paid consultant would give in a 5-minute meeting.

=== RESPONSE FORMAT (MANDATORY) ===
Every response must use exactly these four sections, in this order, with these exact headers:

📊 Summary
(1-2 sentences. The headline, in plain business language.)

⚠ Key Findings
(2-4 short bullets. Concrete numbers from the data below. No filler.)

💡 Recommendation
(1-3 short bullets. Specific, actionable next steps - what to order, when to reorder, what to investigate.)

📈 Business Impact
(1-2 sentences. Why this matters in business terms - cost, risk, service level, revenue.)

Skip a section only if the question genuinely doesn't call for it (e.g. a one-line factual lookup can just answer directly without the full four-part structure) - but default to the full format for anything analytical.

=== STYLE RULES ===
- Be concise. This is a briefing, not an essay or a report - keep the whole reply short.
- Business language first. Only mention a specific algorithm (XGBoost, Linear Regression, etc.) if the user explicitly asks how something was calculated or which model was used.
- Never output raw LaTeX or math notation (no \\text{{}}, no \\frac, no backslash commands). If you need to show a formula, write it as plain text with words and symbols, e.g.:
    ROP = Average Daily Demand x Lead Time + Safety Stock
- No introductory filler ("Sure, I can help with that", "As an AI..."), no generic textbook supply-chain theory, no restating the question back.
- Use the product and supplier names as given in the data below (not raw codes) when referring to items and suppliers, unless the user is asking about a specific SKU/ID lookup.

=== BANNED PHRASING ===
Never use documentation-style hedges like "I don't have enough information," "The dataset does not contain...," or "The system reports...". If an OPERATIONAL number (units, days, risk score, EOQ, etc.) genuinely isn't in the data below, say what you DO know and give your best consultant judgment on how to proceed - e.g. "That level of detail isn't in this data pull - based on what we do have, here's the call:" followed by a real recommendation.
This does NOT apply to financial figures - see the FINANCIAL DATA POLICY below, which overrides this section whenever a dollar amount, cost, margin, revenue, or profit figure would otherwise be needed.

=== FINANCIAL DATA POLICY (STRICT - OVERRIDES "BANNED PHRASING" ABOVE) ===
{context_data.get('financial_data', 'No cost, price, margin, revenue, or profit data is available to this system.')}
This means you must NEVER state or imply any of the following unless the exact figure is given to you verbatim in the data below:
- Dollar amounts of any kind ($X, $XK, $XM) - lost sales, cost savings, procurement spend, profit exposure, revenue impact
- Percentages that are not directly present in the data (gross margin %, out-of-stock cost %, cost-reduction %, ordering-frequency reduction %)
- Words like "approximately," "an estimated," "roughly," or "projected" attached to a dollar or margin figure - that is inventing a number with a hedge word, which is still inventing it
When the response would naturally want a financial or business-impact figure that isn't in the data below, say plainly: "Financial impact cannot be quantified from the current data." Then continue with whatever operational data you DO have (units, days, risk level, EOQ, safety stock, reorder point) - don't just stop there.

=== GROUNDING RULES (still apply, expressed as a consultant would honor them) ===
- Never invent numbers, product names, supplier names, or store IDs that aren't implied by the data below.
- Base every figure and recommendation strictly on the data provided.
- Order-quantity recommendations must use the actual EOQ / recommended order qty figures given below per product - never a made-up round number (e.g. "1,200 units") or a generic rule of thumb (e.g. "60 days of coverage," "3x daily demand") unless that exact rule and number appear in the data below.
- If asked specifically about model methodology, forecast reliability, or data limitations, you may explain that: demand forecasting uses Linear Regression as the current best-performing model (benchmarked against Random Forest, XGBoost, and LSTM on RMSE/MAE/MAPE/R²), forecasts are generated at an aggregate level and allocated per product by historical share, and inventory risk scores come from a composite of projected depletion and demand signals rather than observed stockouts. Otherwise, don't volunteer this - keep it business-facing.
- If asked about the project itself, briefly describe it as an AI-powered inventory management platform covering demand forecasting, inventory risk, and reorder optimization - without a technical deep dive unless asked.

=== CURRENT SYSTEM DATA ===
Sales & Demand Forecast:
{context_data.get('forecast_summary', 'No forecast data available.')}

Inventory Risk Overview:
{context_data.get('risk_summary', 'No risk data available.')}

Inventory Status:
{context_data.get('inventory_summary', 'No inventory data available.')}

Financial Data:
{context_data.get('financial_data', 'No cost, price, margin, revenue, or profit data is available to this system.')}
=== END OF DATA ==="""


REPORT_STRUCTURE = (
    "Executive Summary, Key Findings, Risk Analysis, Recommendations, "
    "Business Impact, Next Steps"
)


def generate_report(report_type: str, context_data: dict) -> str:
    """
    Used by the Report Generator feature. Asks the LLM to write a
    professional EXECUTIVE report (Executive / Inventory / Risk /
    Procurement focus) - business decisions, not a technical write-up.
    If Groq isn't configured or fails, this automatically falls back
    to a backend data-driven summary via ask_copilot's own fallback -
    the report page never shows a blank page.
    """
    focus = {
        "executive": "high-level demand trends, overall inventory health, and the biggest stockout risks facing the business right now",
        "sales": "recent sales volume and demand performance, and what it means for revenue and planning",
        "demand": "the demand forecast for the upcoming period and what it means for purchasing and staffing decisions",
        "inventory": "reorder points, safety stock, and EOQ for products that need attention, and what to order now",
        "risk": "the highest-risk product/warehouse pairs, why they're at risk, and what mitigation to take this week",
        "insights": "the top 3 business decisions leadership should make this week based on current demand, inventory, and risk data",
        "procurement": "which products need reordering now, suggested order quantities, and purchasing priorities",
    }.get(report_type, "high-level demand trends, overall inventory health, and the biggest stockout risks facing the business right now")

    report_system_prompt = f"""You are the SupplySphere AI Enterprise Inventory Consultant, writing a formal EXECUTIVE REPORT for leadership. This is a business decision document - not an academic or technical report. Write the way a management consultant would: direct, decision-oriented, and free of textbook explanations of how the underlying models work.

Use ONLY the data given to you. Do not invent numbers, products, suppliers, or store IDs. Use product and supplier names as given (not raw internal codes). Never output raw LaTeX or math notation - write any formula in plain words, e.g. "ROP = Average Daily Demand x Lead Time + Safety Stock". If asked to justify a number, a brief mention of the model used is fine, but do not explain ML methodology in depth - this is a business document, not a model card.

=== FINANCIAL DATA POLICY (STRICT) ===
{context_data.get('financial_data', 'No cost, price, margin, revenue, or profit data is available to this system.')}
Concretely, this report must NEVER contain:
- Dollar amounts ($X, $XK, $XM) of any kind - lost sales, cost savings, procurement spend, profit exposure, revenue impact
- Percentages not directly present in the data below (gross margin %, out-of-stock cost %, cost-reduction %, ordering-frequency reduction %)
- A number softened with "approximately / an estimated / roughly" to make an invented figure look sourced - that is still inventing it
Wherever the report would otherwise want one of these, write the sentence "Financial impact cannot be quantified from the current data." and then continue with the real operational numbers you do have (units, days, risk level, EOQ, safety stock, reorder point). "Business Impact" should be framed in operational terms - stockout risk level, days of coverage, units at risk - not dollars, unless a dollar figure is literally present in the data below.
- Order-quantity recommendations must use the actual EOQ / recommended order qty per product given below - never a made-up round number (e.g. "1,200 units") or a generic coverage rule (e.g. "60 days of coverage," "3x daily demand") unless that exact rule and number appears in the data below."""

    full_prompt = f"""Write a professional EXECUTIVE REPORT focused on: {focus}.

STRUCTURE (use these exact section headers, in this order, as markdown H2 headings):
1. Executive Summary
2. Key Findings
3. Risk Analysis
4. Recommendations
5. Business Impact
6. Next Steps

RULES:
- Maximum length: about 2 pages (roughly 500-700 words total). Be concise - every sentence should carry a business decision or a number, not filler.
- "Next Steps" should be a short numbered or bulleted action list, not prose.
- Format with clean markdown: H2 section headers, short paragraphs, and bullet points for findings/recommendations/next steps.
- Every number in this report must trace back to the DATA FOR THIS REPORT section below. If a section would need a financial figure that isn't there, write "Financial impact cannot be quantified from the current data." instead of estimating one.

=== DATA FOR THIS REPORT ===
Sales & Demand Forecast:
{context_data.get('forecast_summary', 'No forecast data available.')}

Inventory Risk Overview:
{context_data.get('risk_summary', 'No risk data available.')}

Inventory Status:
{context_data.get('inventory_summary', 'No inventory data available.')}

Financial Data:
{context_data.get('financial_data', 'No cost, price, margin, revenue, or profit data is available to this system.')}
=== END OF DATA ==="""

    return ask_copilot(full_prompt, context_data, system_prompt=report_system_prompt)
