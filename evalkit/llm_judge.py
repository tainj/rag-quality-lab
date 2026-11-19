"""LLM-as-a-judge для negative rejection (unanswerable questions)."""

from functools import lru_cache

from openai import LengthFinishReasonError
from pydantic import BaseModel

from config import LLM_MODEL, NEGATIVE_REJECTION_PROMPT, RAGAS_TIMEOUT
from evalkit.llm_http import openai_client


@lru_cache(maxsize=1)
def _judge_client():
    return openai_client(timeout=RAGAS_TIMEOUT)


def is_declined_verdict(verdict) -> bool:
    """Read the verdict field rather than the truthiness of a non-empty dict."""
    return (
        verdict.get("declined") is True
        if isinstance(verdict, dict)
        else verdict is True
    )


class NegativeRejectionVerdict(BaseModel):
    declined: bool
    reason: str


def judge_negative_rejection(
    question: str, answer: str, model: str = LLM_MODEL
) -> dict:
    try:
        resp = _judge_client().beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": NEGATIVE_REJECTION_PROMPT.format(
                        question=question, answer=answer
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=200,
            response_format=NegativeRejectionVerdict,
        )
        return resp.choices[0].message.parsed.model_dump()
    except LengthFinishReasonError:
        # If model generates too much, assume it's NOT a proper refusal
        return {"declined": False, "reason": "length_limit_reached"}
