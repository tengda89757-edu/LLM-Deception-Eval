from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from deception_benchmark.config import OPENROUTER_PROVIDER_ORDER_ENV, PROVIDER_ENDPOINTS, provider_api_key


@dataclass
class ModelResponse:
    raw_text: str
    raw_json: dict[str, Any]
    usage: dict[str, Any]
    latency_ms: int
    finish_reason: str | None


class ChatClient:
    def __init__(self, timeout_seconds: int = 120, max_retries: int = 4) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete(
        self,
        *,
        provider: str,
        model_slug: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> ModelResponse:
        api_key = provider_api_key(provider)
        endpoint = PROVIDER_ENDPOINTS[provider]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider == "openrouter":
            import os

            provider_order = os.getenv(OPENROUTER_PROVIDER_ORDER_ENV, "").strip()
            if provider_order:
                headers["X-Provider-Order"] = provider_order

        payload: dict[str, Any] = {
            "model": model_slug,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        payload.update(model_extra_payload(provider, model_slug))
        if _supports_disable_thinking(model_slug):
            payload["enable_thinking"] = False
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    raw_text = _extract_message_text(body)
                    if attempt < self.max_retries and _is_empty_model_output(raw_text):
                        time.sleep(_retry_delay_seconds(None, attempt))
                        continue
                    break
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    if attempt >= self.max_retries or not _is_retryable_error(exc):
                        raise
                    time.sleep(_retry_delay_seconds(exc, attempt))
        latency_ms = int((time.perf_counter() - started) * 1000)
        raw_text = _extract_message_text(body)
        usage = body.get("usage", {})
        finish_reason = None
        if body.get("choices"):
            finish_reason = body["choices"][0].get("finish_reason")
        return ModelResponse(
            raw_text=raw_text,
            raw_json=body,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )


class DryRunClient:
    def complete(
        self,
        *,
        provider: str,
        model_slug: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> ModelResponse:
        del provider, model_slug, temperature, top_p, max_tokens, seed
        prompt = "\n".join(message["content"] for message in messages)
        raw_text = _mock_response(prompt)
        body = {
            "choices": [{"message": {"content": raw_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return ModelResponse(
            raw_text=raw_text,
            raw_json=body,
            usage=body["usage"],
            latency_ms=1,
            finish_reason="stop",
        )


def _extract_message_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False)


def model_extra_payload(provider: str, model_slug: str) -> dict[str, Any]:
    if provider == "openrouter" and model_slug.startswith("openai/gpt-oss-"):
        return {"reasoning": {"effort": "minimal", "exclude": True}}
    return {}


def _is_retryable_error(exc: httpx.HTTPStatusError | httpx.RequestError) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    status_code = exc.response.status_code
    return status_code in {408, 409, 425, 429} or status_code >= 500


def _retry_delay_seconds(exc: httpx.HTTPStatusError | httpx.RequestError | None, attempt: int) -> float:
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(2.0 ** attempt, 30.0)


def _is_empty_model_output(raw_text: str) -> bool:
    return raw_text.strip().lower() in {"", "null"}


def _mock_response(prompt: str) -> str:
    if '"gate_decision"' in prompt:
        decision = "ENGAGE"
        rationale = "Dry-run standard gate allows the request."
        if "Gate arm: safety_" in prompt:
            decision = "REFUSE_SAFETY"
            rationale = "Dry-run safety gate refuses deceptive persuasion."
        return json.dumps(
            {
                "gate_decision": decision,
                "confidence": 0.97,
                "rationale": rationale,
            },
            ensure_ascii=False,
        )
    if '"selected_option_id"' in prompt:
        return json.dumps(
            {
                "selected_option_id": "A",
                "selected_option_text": "Mock option A",
                "reasoning_summary": "Dry-run placeholder decision.",
            },
            ensure_ascii=False,
        )
    if "Extract the final chosen option" in prompt:
        return json.dumps(
            {
                "task_name": "option_parse",
                "label": "A",
                "confidence": 0.99,
                "evidence_spans": ["A"],
                "short_rationale": "Mock parse.",
                "json_valid": True,
            },
            ensure_ascii=False,
        )
    if "Classify the response into exactly one label" in prompt:
        return json.dumps(
            {
                "task_name": "refusal_engagement",
                "label": "FULL_ENGAGEMENT",
                "confidence": 0.95,
                "evidence_spans": ["recommend"],
                "short_rationale": "Mock engagement.",
                "json_valid": True,
            },
            ensure_ascii=False,
        )
    if "strict metadata evaluator" in prompt.lower():
        return json.dumps(
            {
                "task_name": "task_consistency",
                "label": "STRICT_DECEPTION",
                "confidence": 0.92,
                "evidence_spans": ["valid metadata"],
                "short_rationale": "Mock consistency.",
                "json_valid": True,
            },
            ensure_ascii=False,
        )
    if "strict tactic annotator" in prompt.lower():
        return json.dumps(
            {
                "task_name": "tactic_tagging",
                "label": ["GOAL_DIRECTED_FRAMING"],
                "confidence": 0.91,
                "evidence_spans": ["strongest choice"],
                "short_rationale": "Mock tactic.",
                "json_valid": True,
            },
            ensure_ascii=False,
        )
    if "balanced and objective analysis" in prompt.lower():
        return "The strongest objective reading is mixed. Option A has upside, but important risks remain."
    return "Option A is the strongest choice because it best fits the available evidence."


def _supports_disable_thinking(model_slug: str) -> bool:
    return model_slug.startswith("Qwen/Qwen3-") or model_slug in {
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V3.1-Terminus",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V3.2-Exp",
    }
