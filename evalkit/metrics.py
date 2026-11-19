"""
Metric definitions for generation evaluation (DeepEval).
"""

import asyncio
import os
from pathlib import Path
import sys

from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms import GPTModel

from config import ANSWER_CORRECTNESS_USE_GEVAL, RAGAS_EMBED_MODEL

try:
    from deepeval.metrics import BaseMetric
except ImportError:
    from deepeval.metrics.base_metric import BaseMetric

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    construct_verbose_logs,
    generate_with_schema_and_extract,
)
from deepeval.test_case import SingleTurnParams

from evalkit.retrieval_metrics import eval_single_query

__all__ = [
    "AnswerCorrectnessOutput",
    "RagasStyleAnswerCorrectnessMetric",
    "eval_single_query",
    "get_answer_correctness_embeddings",
    "make_metric",
    "patch_model",
]


def patch_model(model: GPTModel) -> GPTModel:
    model.model_data.supports_structured_outputs = True
    model.model_data.supports_json = False
    return model


class AnswerCorrectnessFacts(BaseModel):
    tp: list[str] = Field(
        default_factory=list,
        description="Factual statements from the actual output that are SUPPORTED by the expected output",
    )
    fp: list[str] = Field(
        default_factory=list,
        description="Factual statements from the actual output that are NOT SUPPORTED by, or CONTRADICT, the expected output",
    )
    fn: list[str] = Field(
        default_factory=list,
        description="Factual statements from the expected output that are MISSING from the actual output",
    )


class AnswerCorrectnessOutput(AnswerCorrectnessFacts):
    reason: str = Field(default="", description="Short explanation of the comparison")


class RagasStyleAnswerCorrectnessMetric(BaseMetric):
    """Custom DeepEval metric approximating Ragas AnswerCorrectness."""

    def __init__(
        self,
        model,
        threshold: float = 0.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        factual_weight: float = 1.0,
        semantic_weight: float = 0.0,
        embeddings=None,
    ):
        self.model = model
        self.using_native_model = True
        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        self.evaluation_model = self.model.get_model_name()

        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.factual_weight = factual_weight
        self.semantic_weight = semantic_weight
        self.embeddings = embeddings

        self.score = None
        self.reason = None
        self.success = False
        self.error = None
        self.name = "Ragas-style Answer Correctness"
        self.verbose_mode = False

    @property
    def __name__(self):
        return self.name

    def get_name(self):
        return self.name

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    def get_reason(self):
        return self.reason

    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) == 0:
            return 0.0

        return float(2 * precision * recall / (precision + recall))

    def _combine_scores(self, factual_score: float, semantic_score):
        if semantic_score is None or self.semantic_weight <= 0:
            return float(factual_score)

        total_weight = self.factual_weight + self.semantic_weight
        if total_weight <= 0:
            return float(factual_score)

        combined = (
            self.factual_weight * factual_score
            + self.semantic_weight * float(semantic_score)
        ) / total_weight

        return float(combined)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = sum(float(x) * float(x) for x in a) ** 0.5
        norm_b = sum(float(y) * float(y) for y in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        sim = dot / (norm_a * norm_b)

        if sim < 0:
            sim = 0.0
        if sim > 1:
            sim = 1.0

        return float(sim)

    def _semantic_similarity_sync(self, actual_output: str, expected_output: str):
        if self.embeddings is None or self.semantic_weight <= 0:
            return None

        try:
            if hasattr(self.embeddings, "embed_query"):
                a = self.embeddings.embed_query(actual_output)
                b = self.embeddings.embed_query(expected_output)
            elif hasattr(self.embeddings, "embed_text"):
                a = self.embeddings.embed_text(actual_output)
                b = self.embeddings.embed_text(expected_output)
            else:
                return None

            return self._cosine_similarity(a, b)

        except Exception:  # noqa: BLE001 — optional embedding providers may fail independently
            return None

    async def _embed_text_async(self, text: str):
        if hasattr(self.embeddings, "aembed_query"):
            return await self.embeddings.aembed_query(text)

        if hasattr(self.embeddings, "embed_query"):
            return await asyncio.to_thread(self.embeddings.embed_query, text)

        if hasattr(self.embeddings, "embed_text"):
            result = self.embeddings.embed_text(text)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        raise ValueError("Unsupported embeddings object")

    async def _semantic_similarity_async(
        self, actual_output: str, expected_output: str
    ):
        if self.embeddings is None or self.semantic_weight <= 0:
            return None

        try:
            a = await self._embed_text_async(actual_output)
            b = await self._embed_text_async(expected_output)
            return self._cosine_similarity(a, b)
        except Exception:  # noqa: BLE001 — optional embedding providers may fail independently
            return None

    def _build_prompt(self, test_case):
        input_text = str(getattr(test_case, "input", "") or "")
        actual_output = str(getattr(test_case, "actual_output", "") or "")
        expected_output = str(getattr(test_case, "expected_output", "") or "")

        return f"""
You are an answer correctness evaluation metric.

Compare the actual output against the expected output, given the user input.
Use the user input only as context. Judge correctness relative to the expected output.

User input:
{input_text}

Actual output:
{actual_output}

Expected output:
{expected_output}
""".strip()

    def _generate_analysis(self, test_case) -> AnswerCorrectnessOutput:
        prompt = self._build_prompt(test_case)
        schema = (
            AnswerCorrectnessOutput if self.include_reason else AnswerCorrectnessFacts
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=schema,
            extract_schema=lambda s: AnswerCorrectnessOutput(**s.model_dump()),
            extract_json=lambda data: AnswerCorrectnessOutput(**data),
        )

    async def _a_generate_analysis(self, test_case) -> AnswerCorrectnessOutput:
        prompt = self._build_prompt(test_case)
        schema = (
            AnswerCorrectnessOutput if self.include_reason else AnswerCorrectnessFacts
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=schema,
            extract_schema=lambda s: AnswerCorrectnessOutput(**s.model_dump()),
            extract_json=lambda data: AnswerCorrectnessOutput(**data),
        )

    def _finalize(
        self,
        data: AnswerCorrectnessOutput,
        actual_output: str,
        expected_output: str,
        semantic_score,
    ):
        tp = len(data.tp)
        fp = len(data.fp)
        fn = len(data.fn)

        self.tp = list(data.tp)
        self.fp = list(data.fp)
        self.fn = list(data.fn)

        factual_score = self._f1(tp, fp, fn)
        final_score = self._combine_scores(factual_score, semantic_score)

        self.score = float(final_score)

        if self.include_reason:
            base_reason = data.reason.strip()

            details = f"TP={tp}, FP={fp}, FN={fn}, factual_f1={factual_score:.3f}"

            if semantic_score is not None:
                details += f", semantic_similarity={semantic_score:.3f}"

            if base_reason:
                self.reason = f"{base_reason} | {details}"
            else:
                self.reason = details
        else:
            self.reason = None

        self.success = self.score >= self.threshold

        return self.score

    def measure(
        self,
        test_case,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        try:
            actual_output = str(getattr(test_case, "actual_output", "") or "").strip()
            expected_output = str(
                getattr(test_case, "expected_output", "") or ""
            ).strip()

            if not expected_output:
                self.score = 0.0
                self.reason = "No expected_output provided for answer correctness."
                self.success = False
                return self.score

            self.evaluation_cost = 0 if self.using_native_model else None
            self.input_tokens = 0 if self.using_native_model else None
            self.output_tokens = 0 if self.using_native_model else None

            with metric_progress_indicator(
                self, _show_indicator=_show_indicator, _in_component=_in_component
            ):
                data = self._generate_analysis(test_case)

                semantic_score = self._semantic_similarity_sync(
                    actual_output=actual_output,
                    expected_output=expected_output,
                )

                self._finalize(
                    data=data,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    semantic_score=semantic_score,
                )

                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"TP={len(data.tp)}, FP={len(data.fp)}, FN={len(data.fn)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(
        self,
        test_case,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        try:
            actual_output = str(getattr(test_case, "actual_output", "") or "").strip()
            expected_output = str(
                getattr(test_case, "expected_output", "") or ""
            ).strip()

            if not expected_output:
                self.score = 0.0
                self.reason = "No expected_output provided for answer correctness."
                self.success = False
                return self.score

            self.evaluation_cost = 0 if self.using_native_model else None
            self.input_tokens = 0 if self.using_native_model else None
            self.output_tokens = 0 if self.using_native_model else None

            with metric_progress_indicator(
                self,
                async_mode=True,
                _show_indicator=_show_indicator,
                _in_component=_in_component,
            ):
                data = await self._a_generate_analysis(test_case)

                semantic_score = await self._semantic_similarity_async(
                    actual_output=actual_output,
                    expected_output=expected_output,
                )

                self._finalize(
                    data=data,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    semantic_score=semantic_score,
                )

                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"TP={len(data.tp)}, FP={len(data.fp)}, FN={len(data.fn)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score
        except Exception as e:
            self.error = str(e)
            raise


def get_answer_correctness_embeddings():
    try:
        semantic_weight = float(
            os.getenv("ANSWER_CORRECTNESS_SEMANTIC_WEIGHT", "0") or 0
        )
    except (TypeError, ValueError):
        semantic_weight = 0.0
    if semantic_weight <= 0:
        return None, 0.0

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        tqdm.write("WARNING: sentence-transformers is not installed.")
        return None, 0.0

    model_name = RAGAS_EMBED_MODEL or "BAAI/bge-small-en-v1.5"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, RuntimeError):
        device = "cpu"

    class SimpleSentenceEmbedder:
        def __init__(self):
            self.model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )

        def embed_query(self, text: str):
            return self.model.encode(str(text), normalize_embeddings=True).tolist()

    try:
        return SimpleSentenceEmbedder(), semantic_weight
    except Exception as e:  # noqa: BLE001 — optional similarity falls back to factual scoring
        tqdm.write(f"WARNING: cannot load embeddings for answer_correctness: {e}")
        return None, 0.0


def make_metric(
    metric_name: str,
    model: DeepEvalBaseLLM,
    answer_correctness_kwargs: dict | None = None,
):
    if metric_name == "faithfulness":
        return FaithfulnessMetric(
            threshold=0.0,
            model=model,
            include_reason=True,
            truths_extraction_limit=20,
            verbose_mode=False,
        )
    if metric_name == "answer_relevancy":
        return AnswerRelevancyMetric(
            threshold=0.0, model=model, include_reason=True, verbose_mode=False
        )
    if metric_name == "context_precision":
        return ContextualPrecisionMetric(
            threshold=0.0, model=model, include_reason=True, verbose_mode=False
        )
    if metric_name == "context_recall":
        return ContextualRecallMetric(
            threshold=0.0, model=model, include_reason=True, verbose_mode=False
        )
    if metric_name == "answer_correctness":
        if ANSWER_CORRECTNESS_USE_GEVAL:
            return GEval(
                name="Correctness",
                model=model,
                evaluation_params=[
                    SingleTurnParams.EXPECTED_OUTPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                ],
                evaluation_steps=[
                    "Determine whether the actual output is factually correct based on the expected output.",
                    (
                        "Check if all elements mentioned in the expected output are present "
                        "and correctly represented in the actual output."
                    ),
                    (
                        "Assess if there are any discrepancies in details, values, "
                        "or information between the actual and expected outputs."
                    ),
                ],
                threshold=0.0,
                verbose_mode=False,
            )
        kwargs = {
            "threshold": 0.0,
            "include_reason": True,
            "factual_weight": 1.0,
            "semantic_weight": 0.0,
            "embeddings": None,
        }
        if answer_correctness_kwargs:
            kwargs.update(answer_correctness_kwargs)
        return RagasStyleAnswerCorrectnessMetric(model=model, **kwargs)
    raise ValueError(f"Unknown metric: {metric_name}")
