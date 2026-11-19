"""LLM judging and token-aware completion utilities for the agent-core pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re

from config import (
    JUDGE_INCLUDE_REASON,
    JUDGE_MAX_RETRIES,
    JUDGE_MAX_TOKENS,
    JUDGE_TIMEOUT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_TEMPERATURE,
    RAG_SYSTEM_PROMPT,
    RAGAS_METRICS,
    RAGAS_TIMEOUT,
)
from evalkit.usage import normalize_usage

# deepeval metrics available in evalkit.metrics.make_metric
DEEPEVAL_METRICS = (
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

DECLINE_TEXT = "Информации недостаточно для ответа."
DECLINE_MARKERS = ("недостаточно", "нет информации", "не найден", "не содержит")

ANSWER_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT  # compatibility alias

_WORD = re.compile(r"\w+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?…])\s+|\n+")


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 2]


def is_declined(answer: str) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in DECLINE_MARKERS)


def _context_block(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))


class _OpenAIMixin:
    def __init__(self, model: str = LLM_MODEL):
        from evalkit.llm_http import openai_client

        self.model = model
        self.client = openai_client(timeout=RAGAS_TIMEOUT)

    def _completion(
        self,
        messages: list[dict],
        max_tokens: int = LLM_MAX_TOKENS,
        model: str | None = None,
    ):
        return self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=max_tokens,
            **(
                {"reasoning_effort": LLM_REASONING_EFFORT}
                if LLM_REASONING_EFFORT is not None
                else {}
            ),
        )

    def _chat(self, messages: list[dict], max_tokens: int = LLM_MAX_TOKENS) -> str:
        resp = self._completion(messages, max_tokens)
        return resp.choices[0].message.content or ""


class LLMAnswerer(_OpenAIMixin):
    name = "llm"

    def _messages(
        self, question: str, contexts: list[str], system_prompt: str | None = None
    ) -> list[dict]:
        context = _context_block(contexts) or "(ничего не найдено)"
        return [
            {
                "role": "system",
                "content": system_prompt
                if system_prompt is not None
                else RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос: {question}",
            },
        ]

    def answer(
        self, question: str, contexts: list[str], system_prompt: str | None = None
    ) -> str:
        return self._chat(self._messages(question, contexts, system_prompt))

    def generate(
        self,
        question: str,
        contexts: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> dict:
        response = self._completion(
            self._messages(question, contexts, system_prompt), model=model
        )
        return {
            "answer": response.choices[0].message.content or "",
            "usage": normalize_usage(response.usage),
        }


class DeepEvalJudge:
    """LLM-as-a-judge through deepeval (same metrics as notebook 04).

    Answerable questions: every metric from RAGAS_METRICS that deepeval
    implements (see DEEPEVAL_METRICS), run in parallel. Unanswerable ones:
    did the system decline (evalkit.llm_judge.judge_negative_rejection).
    A metric that fails is returned as None, with the error under "errors".
    """

    name = "deepeval"
    supports_metric_retry = True

    def __init__(self, metrics: list[str] | None = None):
        from deepeval.models.llms import GPTModel

        from evalkit.llm_http import http_clients
        from evalkit.metrics import patch_model

        self.metric_names = [
            m for m in (metrics or RAGAS_METRICS) if m in DEEPEVAL_METRICS
        ]
        self.model = patch_model(
            GPTModel(
                model=LLM_MODEL.removeprefix("openai/"),
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                generation_kwargs={
                    "max_tokens": JUDGE_MAX_TOKENS,
                    **(
                        {"reasoning_effort": LLM_REASONING_EFFORT}
                        if LLM_REASONING_EFFORT is not None
                        else {}
                    ),
                },
                timeout=JUDGE_TIMEOUT,
                max_retries=JUDGE_MAX_RETRIES,
                **http_clients(JUDGE_TIMEOUT),
            )
        )

    def _measure(self, name: str, test_case) -> tuple[float | None, str | None]:
        from evalkit.metrics import make_metric

        metric = make_metric(name, model=self.model)
        # sync calls: the metrics run in parallel threads, and an async http
        # client cannot be shared between their event loops
        metric.async_mode = False
        # Only scores are persisted by this pipeline; avoid auxiliary LLM
        # explanation requests that can fail after the score is calculated.
        metric.include_reason = JUDGE_INCLUDE_REASON
        try:
            metric.measure(test_case, _show_indicator=False)
            return metric.score, metric.error
        except Exception as e:  # noqa: BLE001 — one metric must not kill the rest
            return None, f"{type(e).__name__}: {e}"

    def score(
        self,
        question: str,
        answer: str,
        reference: str,
        contexts: list[str],
        unanswerable: bool = False,
        *,
        metrics: list[str] | None = None,
    ) -> dict:
        if unanswerable:
            from evalkit.llm_judge import is_declined_verdict, judge_negative_rejection

            verdict = judge_negative_rejection(question, answer)
            return {"declined": float(is_declined_verdict(verdict))}

        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=question,
            actual_output=answer or "(no answer)",
            expected_output=reference or "(no reference)",
            retrieval_context=contexts or ["No retrieval context provided."],
        )
        names = self.metric_names if metrics is None else metrics
        if set(names) - set(self.metric_names):
            raise ValueError("Requested metric is not enabled for this judge")
        with ThreadPoolExecutor(max_workers=len(names) or 1) as pool:
            results = dict(
                zip(
                    names,
                    pool.map(lambda m: self._measure(m, test_case), names),
                )
            )
        scores: dict = {m: score for m, (score, _) in results.items()}
        errors = {m: err for m, (_, err) in results.items() if err}
        if errors:
            scores["errors"] = errors
        scores["declined"] = float(is_declined(answer))
        return scores


def make_judge(mode: str = "real"):
    if mode != "real":
        raise ValueError("Only real LLM judging is supported")
    return DeepEvalJudge()
