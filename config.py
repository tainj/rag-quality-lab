"""Compatibility imports for notebooks and evalkit; edit configs/settings.json.

RAG_EVAL_CONFIG selects a complete alternative JSON settings file.
Secrets are read exclusively from environment variables or PROJECT_ROOT/.env.
"""

import os
from pathlib import Path

from evalkit.json_config import env_bool, load_json_object, require_setting

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("RAG_EVAL_ROOT", REPO_ROOT)).expanduser().resolve()
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _key, _sep, _value = _line.strip().partition("=")
        _key = _key.strip()
        if _sep and _key and not _key.startswith("#"):
            os.environ.setdefault(_key, _value.strip().strip("\"'"))

CONFIG_PATH = Path(
    os.environ.get("RAG_EVAL_CONFIG", REPO_ROOT / "configs/settings.json")
).expanduser()
_SETTINGS = load_json_object(CONFIG_PATH)
if {"AGENT_PLATFORM_TOKEN", "LLM_API_KEY"} & _SETTINGS.keys():
    raise ValueError("Keep AGENT_PLATFORM_TOKEN and LLM_API_KEY in .env, not JSON")


def _setting(name, expected):
    return require_setting(_SETTINGS, name, expected)


def _path(name):
    path = Path(_setting(name, str)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


RAG_EVAL_AGENT_ID = _setting("RAG_EVAL_AGENT_ID", str)
AGENT_WORKER_POOL_SIZE = _setting("AGENT_WORKER_POOL_SIZE", int)
LLM_MODEL = os.environ.get("LLM_MODEL", _setting("LLM_MODEL", str))
LLM_TEMPERATURE = _setting("LLM_TEMPERATURE", (int, float))
LLM_MAX_TOKENS = _setting("LLM_MAX_TOKENS", int)
TESTSET_SIZE = _setting("TESTSET_SIZE", int)
QUERY_DISTRIBUTION = _setting("QUERY_DISTRIBUTION", dict)
EMBEDDING_MODEL_PRESETS = _setting("EMBEDDING_MODEL_PRESETS", dict)
PIPELINE_DEV_FRACTION = _setting("PIPELINE_DEV_FRACTION", (int, float))
PIPELINE_SELECT_METRIC = _setting("PIPELINE_SELECT_METRIC", str)
PIPELINE_TOP_N_GENERATION = _setting("PIPELINE_TOP_N_GENERATION", int)
PIPELINE_GENERATION_SAMPLE = _setting("PIPELINE_GENERATION_SAMPLE", int)
PIPELINE_REAL_SAMPLE = _setting("PIPELINE_REAL_SAMPLE", int)
PIPELINE_GENERATION_TOP_K = _setting("PIPELINE_GENERATION_TOP_K", int)
EVAL_COLLECTION_NAME_PREFIX = _setting("EVAL_COLLECTION_NAME_PREFIX", str)
EVAL_COLLECTION_EMBEDDING = _setting("EVAL_COLLECTION_EMBEDDING", dict)
EVAL_COLLECTION_CHUNKING = _setting("EVAL_COLLECTION_CHUNKING", dict)
EVAL_COLLECTION_RERANKER = _setting("EVAL_COLLECTION_RERANKER", dict)
EVAL_DATASETS = _setting("EVAL_DATASETS", dict)
RAGAS_METRICS = _setting("RAGAS_METRICS", list)
RAGAS_EMBED_MODEL = _setting("RAGAS_EMBED_MODEL", str)
RAGAS_NUM_WORKERS = _setting("RAGAS_NUM_WORKERS", int)
RAGAS_TIMEOUT = _setting("RAGAS_TIMEOUT", int)
JUDGE_MAX_TOKENS = _SETTINGS.get("JUDGE_MAX_TOKENS", LLM_MAX_TOKENS)
JUDGE_TIMEOUT = _SETTINGS.get("JUDGE_TIMEOUT", RAGAS_TIMEOUT)
JUDGE_INCLUDE_REASON = _SETTINGS.get("JUDGE_INCLUDE_REASON", True)
if type(JUDGE_INCLUDE_REASON) is not bool:
    raise ValueError("JUDGE_INCLUDE_REASON must be boolean")
for _judge_name, _judge_value in (
    ("JUDGE_MAX_TOKENS", JUDGE_MAX_TOKENS),
    ("JUDGE_TIMEOUT", JUDGE_TIMEOUT),
):
    if type(_judge_value) is not int or _judge_value <= 0:
        raise ValueError(f"{_judge_name} must be a positive integer")
RAGAS_EVAL_SAMPLE_FRACTION = _setting("RAGAS_EVAL_SAMPLE_FRACTION", (int, float))
ANSWER_CORRECTNESS_USE_GEVAL = _setting("ANSWER_CORRECTNESS_USE_GEVAL", bool)
TOP_K = _setting("TOP_K", int)
K_SET = tuple(_setting("K_SET", list))
RELEVANCE_THRESHOLD = _setting("RELEVANCE_THRESHOLD", (int, float))
RELEVANCE_METRIC = _setting("RELEVANCE_METRIC", str)
DEDUP_SIMILARITY_THRESHOLD = _setting("DEDUP_SIMILARITY_THRESHOLD", (int, float))
LEAKAGE_WORD_OVERLAP_THRESHOLD = _setting(
    "LEAKAGE_WORD_OVERLAP_THRESHOLD", (int, float)
)
UNANSWERABLE_SYSTEM_PROMPT = _setting("UNANSWERABLE_SYSTEM_PROMPT", str)
EXACT_MATCH_SYSTEM_PROMPT = _setting("EXACT_MATCH_SYSTEM_PROMPT", str)
SINGLE_HOP_PROMPT = _setting("SINGLE_HOP_PROMPT", str)
MULTI_HOP_PROMPT = _setting("MULTI_HOP_PROMPT", str)
UNANSWERABLE_PROMPT = _setting("UNANSWERABLE_PROMPT", str)
EXACT_MATCH_PROMPT = _setting("EXACT_MATCH_PROMPT", str)
DEDUP_JUDGE_PROMPT = _setting("DEDUP_JUDGE_PROMPT", str)
FINAL_FORMAT_VERSION = _setting("FINAL_FORMAT_VERSION", str)
TARGET_VALIDATED_SAMPLES = _setting("TARGET_VALIDATED_SAMPLES", int)
CSV_PATH = _path("CSV_PATH")
DATA_DIR = _path("DATA_DIR")
RUNS_DIR = _path("RUNS_DIR")
AGENT_PLATFORM_URL = os.environ.get(
    "AGENT_PLATFORM_URL", _setting("AGENT_PLATFORM_URL", str)
)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", _setting("LLM_BASE_URL", str))
LLM_VIA_PLATFORM = env_bool(
    "LLM_VIA_PLATFORM", _setting("LLM_VIA_PLATFORM", bool), os.environ
)
AGENT_DESCRIPTION = _setting("AGENT_DESCRIPTION", str)
RAG_TOOL_DESCRIPTION = _setting("RAG_TOOL_DESCRIPTION", str)
RAG_SYSTEM_PROMPT = _setting("RAG_SYSTEM_PROMPT", str)
NEGATIVE_REJECTION_PROMPT = _setting("NEGATIVE_REJECTION_PROMPT", str)
AGENT_TOOL_PROMPT = _setting("AGENT_TOOL_PROMPT", str)

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
AGENT_MODEL = os.environ.get("AGENT_MODEL", _setting("AGENT_MODEL", str))
AGENT_PLATFORM_TOKEN = os.environ.get("AGENT_PLATFORM_TOKEN", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
if LLM_VIA_PLATFORM:
    LLM_BASE_URL = AGENT_PLATFORM_URL
    LLM_API_KEY = AGENT_PLATFORM_TOKEN

if not 0 < PIPELINE_DEV_FRACTION < 1:
    raise ValueError("PIPELINE_DEV_FRACTION must be between 0 and 1")
if not K_SET or any(type(k) is not int or k <= 0 for k in K_SET):
    raise ValueError("K_SET must contain positive integers")
if RELEVANCE_METRIC not in ("coverage", "iou") or not 0 < RELEVANCE_THRESHOLD <= 1:
    raise ValueError("Invalid relevance metric or threshold")
if not RAG_SYSTEM_PROMPT.strip():
    raise ValueError("RAG_SYSTEM_PROMPT must not be blank")
if "{tool_name}" not in AGENT_TOOL_PROMPT:
    raise ValueError("AGENT_TOOL_PROMPT must contain {tool_name}")

LLM_REASONING_EFFORT = _SETTINGS.get("LLM_REASONING_EFFORT")
LLM_MAX_RETRIES = _SETTINGS.get("LLM_MAX_RETRIES", 2)
if LLM_REASONING_EFFORT not in (
    None,
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
):
    raise ValueError("Invalid LLM_REASONING_EFFORT")
if type(LLM_MAX_RETRIES) is not int or LLM_MAX_RETRIES < 0:
    raise ValueError("LLM_MAX_RETRIES must be a non-negative integer")
JUDGE_MAX_RETRIES = _SETTINGS.get("JUDGE_MAX_RETRIES", LLM_MAX_RETRIES)
if type(JUDGE_MAX_RETRIES) is not int or JUDGE_MAX_RETRIES < 0:
    raise ValueError("JUDGE_MAX_RETRIES must be a non-negative integer")

SYNTHESIS_DEFAULT_SIZE = _setting("SYNTHESIS_DEFAULT_SIZE", int)

SYNTHESIS_MAX_SOURCE_CHARS = _setting("SYNTHESIS_MAX_SOURCE_CHARS", int)

SYNTHESIS_GOLD_MAX_CHARS = _setting("SYNTHESIS_GOLD_MAX_CHARS", int)

SYNTHESIS_MAX_ATTEMPTS_FACTOR = _setting("SYNTHESIS_MAX_ATTEMPTS_FACTOR", int)

SYNTHESIS_SYSTEM_PROMPT = _setting("SYNTHESIS_SYSTEM_PROMPT", str)
