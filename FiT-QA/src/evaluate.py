#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate prediction JSONL with corpus BLEU and LLM-as-a-judge.

This script merges:
1) prediction JSONL (must include `answer` by default), and
2) QA JSONL (must include `question` and `answer`),

then computes:
- corpus BLEU (sacrebleu)
- LLM-as-a-judge score using Qwen/Qwen3-VL-8B-Instruct (1-5)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sacrebleu


DEFAULT_ANSWER_KEYS: Tuple[str, ...] = ("answer",)
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_JUDGE_TEMPLATE = (
    "You are an expert evaluator.\n"
    "Evaluate how well the Prediction matches the Answer for the Question.\n\n"
    "Scoring rubric:\n"
    "5: Fully correct and complete.\n"
    "4: Mostly correct with minor errors.\n"
    "3: Partially correct.\n"
    "2: Mostly incorrect.\n"
    "1: Completely incorrect or irrelevant.\n\n"
    "Output exactly one integer from 1 to 5. No explanation.\n\n"
    "Question: {Question}\n"
    "Answer: {Answer}\n"
    "Prediction: {Prediction}\n\n"
    "Score:\n"
)

FULLWIDTH_TO_ASCII = str.maketrans("０１２３４５６７８９", "0123456789")
JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
ENGLISH_SCORE_WORDS: Dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
LABEL_SCORE_RULES: Tuple[Tuple[str, int], ...] = (
    ("very poor", 1),
    ("poor", 2),
    ("fair", 3),
    ("good", 4),
    ("excellent", 5),
)


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    question: str
    reference: str
    prediction: str
    prediction_line: int
    qa_line: int


@dataclass(frozen=True)
class PairingStats:
    total_prediction_rows: int
    total_qa_rows: int
    used_samples: int
    skipped_status: int
    skipped_missing_prediction: int
    skipped_missing_question: int
    skipped_missing_answer: int
    skipped_missing_match: int
    extra_unmatched_qa_rows: int
    pairing_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "prediction JSONL と QA JSONL を突合し、"
            "corpus BLEU と Qwen3-VL judge score を計算します。"
        )
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        required=True,
        help="prediction を含む JSONL ファイル",
    )
    parser.add_argument(
        "--qa-jsonl",
        type=Path,
        required=True,
        help="question / answer を含む JSONL ファイル (例: test.jsonl)",
    )
    parser.add_argument(
        "--prediction-key",
        type=str,
        default="answer",
        help="prediction 側の予測テキストキー (default: answer)",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="QA 側の質問テキストキー (default: question)",
    )
    parser.add_argument(
        "--answer-keys",
        type=str,
        default=",".join(DEFAULT_ANSWER_KEYS),
        help="QA 側の正解候補キーをカンマ区切りで指定 (default: answer)",
    )
    parser.add_argument(
        "--match-key",
        type=str,
        default=None,
        help="2つのJSONLをキー結合する場合の共通キー。未指定なら行番号順で突合",
    )
    parser.add_argument(
        "--pred-match-key",
        type=str,
        default=None,
        help="prediction 側の結合キー名 (未指定時は --match-key)",
    )
    parser.add_argument(
        "--qa-match-key",
        type=str,
        default=None,
        help="QA 側の結合キー名 (未指定時は --match-key)",
    )
    parser.add_argument(
        "--include-non-ok",
        action="store_true",
        help="prediction 側で status != ok の行も評価対象に含める",
    )
    parser.add_argument(
        "--disable-mecab",
        action="store_true",
        help="MeCab 分かち書きを使わず BLEU を計算する",
    )
    parser.add_argument(
        "--mecab-args",
        type=str,
        default="-Owakati",
        help="MeCab.Tagger に渡す引数 (default: -Owakati)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="LLM judge に使う Hugging Face モデル",
    )
    parser.add_argument(
        "--judge-max-samples",
        type=int,
        default=None,
        help="judge の最大評価件数 (先頭N件)。未指定は全件",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=16,
        help="judge 生成時の max_new_tokens (default: 16)",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.2,
        help="judge 生成時の temperature (default: 0.2)",
    )
    parser.add_argument(
        "--judge-top-p",
        type=float,
        default=0.95,
        help="judge 生成時の top_p (default: 0.95)",
    )
    parser.add_argument(
        "--judge-top-k",
        type=int,
        default=40,
        help="judge 生成時の top_k (default: 40)",
    )
    parser.add_argument(
        "--judge-device-map",
        type=str,
        default="auto",
        help="Hugging Face model の device_map (default: auto)",
    )
    parser.add_argument(
        "--judge-torch-dtype",
        type=str,
        default="auto",
        help="auto / bf16 / fp16 / fp32",
    )
    parser.add_argument(
        "--judge-attn-implementation",
        type=str,
        default=None,
        help="attn_implementation を明示する場合に指定",
    )
    parser.add_argument(
        "--fallback-score-on-parse-failure",
        type=int,
        default=3,
        help="judge 出力のスコア抽出失敗時に使う救済スコア (1..5, default: 3)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="結果を保存する JSON パス",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse failed: {path}:{line_no}", file=sys.stderr)
                continue
            if isinstance(row, dict):
                row["_line_no"] = line_no
                rows.append(row)
            else:
                print(f"[WARN] Non-object JSON ignored: {path}:{line_no}", file=sys.stderr)
    return rows


def read_answer_keys(raw: str) -> List[str]:
    keys = [x.strip() for x in raw.split(",") if x.strip()]
    return keys or list(DEFAULT_ANSWER_KEYS)


def pick_first_present_text(row: dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def status_is_ok(row: dict[str, Any]) -> bool:
    status = row.get("status")
    return status in (None, "ok")


def build_eval_samples(
    prediction_rows: Sequence[dict[str, Any]],
    qa_rows: Sequence[dict[str, Any]],
    prediction_key: str,
    question_key: str,
    answer_keys: Sequence[str],
    include_non_ok: bool,
    match_key: Optional[str],
    pred_match_key: Optional[str],
    qa_match_key: Optional[str],
) -> tuple[List[EvalSample], PairingStats]:
    skipped_status = 0
    skipped_missing_prediction = 0
    skipped_missing_question = 0
    skipped_missing_answer = 0
    skipped_missing_match = 0

    samples: List[EvalSample] = []

    if match_key:
        prediction_match_key = pred_match_key or match_key
        qa_match_key_resolved = qa_match_key or match_key

        qa_index: Dict[str, dict[str, Any]] = {}
        for row in qa_rows:
            key_value = row.get(qa_match_key_resolved)
            if key_value is None:
                continue
            key_str = str(key_value)
            if key_str not in qa_index:
                qa_index[key_str] = row

        matched_qa_keys: set[str] = set()
        for pred_row in prediction_rows:
            if (not include_non_ok) and (not status_is_ok(pred_row)):
                skipped_status += 1
                continue

            pred_text = pred_row.get(prediction_key)
            if pred_text is None or not str(pred_text).strip():
                skipped_missing_prediction += 1
                continue

            key_value = pred_row.get(prediction_match_key)
            if key_value is None:
                skipped_missing_match += 1
                continue
            key_str = str(key_value)

            qa_row = qa_index.get(key_str)
            if qa_row is None:
                skipped_missing_match += 1
                continue

            question = qa_row.get(question_key)
            answer = pick_first_present_text(qa_row, answer_keys)
            if question is None or not str(question).strip():
                skipped_missing_question += 1
                continue
            if answer is None:
                skipped_missing_answer += 1
                continue

            samples.append(
                EvalSample(
                    sample_id=key_str,
                    question=str(question).strip(),
                    reference=answer,
                    prediction=str(pred_text).strip(),
                    prediction_line=int(pred_row.get("_line_no", -1)),
                    qa_line=int(qa_row.get("_line_no", -1)),
                )
            )
            matched_qa_keys.add(key_str)

        extra_unmatched_qa_rows = max(0, len(qa_index) - len(matched_qa_keys))
        stats = PairingStats(
            total_prediction_rows=len(prediction_rows),
            total_qa_rows=len(qa_rows),
            used_samples=len(samples),
            skipped_status=skipped_status,
            skipped_missing_prediction=skipped_missing_prediction,
            skipped_missing_question=skipped_missing_question,
            skipped_missing_answer=skipped_missing_answer,
            skipped_missing_match=skipped_missing_match,
            extra_unmatched_qa_rows=extra_unmatched_qa_rows,
            pairing_mode="key",
        )
        return samples, stats

    pair_len = min(len(prediction_rows), len(qa_rows))
    for idx in range(pair_len):
        pred_row = prediction_rows[idx]
        qa_row = qa_rows[idx]

        if (not include_non_ok) and (not status_is_ok(pred_row)):
            skipped_status += 1
            continue

        pred_text = pred_row.get(prediction_key)
        if pred_text is None or not str(pred_text).strip():
            skipped_missing_prediction += 1
            continue

        question = qa_row.get(question_key)
        if question is None or not str(question).strip():
            skipped_missing_question += 1
            continue

        answer = pick_first_present_text(qa_row, answer_keys)
        if answer is None:
            skipped_missing_answer += 1
            continue

        sample_id = str(pred_row.get("sample_id", "")).strip() or str(idx)
        samples.append(
            EvalSample(
                sample_id=sample_id,
                question=str(question).strip(),
                reference=answer,
                prediction=str(pred_text).strip(),
                prediction_line=int(pred_row.get("_line_no", -1)),
                qa_line=int(qa_row.get("_line_no", -1)),
            )
        )

    skipped_missing_match = abs(len(prediction_rows) - len(qa_rows))
    extra_unmatched_qa_rows = max(0, len(qa_rows) - len(prediction_rows))
    stats = PairingStats(
        total_prediction_rows=len(prediction_rows),
        total_qa_rows=len(qa_rows),
        used_samples=len(samples),
        skipped_status=skipped_status,
        skipped_missing_prediction=skipped_missing_prediction,
        skipped_missing_question=skipped_missing_question,
        skipped_missing_answer=skipped_missing_answer,
        skipped_missing_match=skipped_missing_match,
        extra_unmatched_qa_rows=extra_unmatched_qa_rows,
        pairing_mode="line",
    )
    return samples, stats


def build_mecab_tagger(mecab_args: str) -> Any:
    try:
        import MeCab
    except ImportError as exc:
        raise RuntimeError(
            "MeCab が見つかりません。`pip install mecab-python3 unidic-lite` を実行してください。"
        ) from exc

    candidates: List[str] = [mecab_args]
    if "-d" not in mecab_args:
        try:
            import unidic_lite

            candidates.append(f"{mecab_args} -d {unidic_lite.DICDIR}")
        except ImportError:
            pass

    last_exc: RuntimeError | None = None
    for candidate in candidates:
        try:
            return MeCab.Tagger(candidate)
        except RuntimeError as exc:
            last_exc = exc

    raise RuntimeError(
        "MeCab Tagger の初期化に失敗しました。辞書設定を見直してください。"
    ) from last_exc


def mecab_tokenize(tagger: Any, text: str) -> str:
    parsed = tagger.parse(text)
    if parsed is None:
        return ""
    return " ".join(parsed.strip().split())


def compute_corpus_bleu(
    samples: Sequence[EvalSample],
    use_mecab: bool,
    mecab_args: str,
) -> dict[str, Any]:
    tokenized_pairs: List[Tuple[str, str]] = []
    skipped_empty = 0

    tagger = build_mecab_tagger(mecab_args) if use_mecab else None
    for sample in samples:
        pred = sample.prediction.strip()
        ref = sample.reference.strip()
        if tagger is not None:
            pred = mecab_tokenize(tagger, pred)
            ref = mecab_tokenize(tagger, ref)
        if not pred or not ref:
            skipped_empty += 1
            continue
        tokenized_pairs.append((pred, ref))

    if not tokenized_pairs:
        return {
            "bleu": 0.0,
            "bp": 0.0,
            "sys_len": 0,
            "ref_len": 0,
            "precisions": [0.0, 0.0, 0.0, 0.0],
            "used_samples": 0,
            "skipped_empty_after_tokenize": skipped_empty,
            "tokenizer": "mecab" if use_mecab else "none",
        }

    hypotheses = [x[0] for x in tokenized_pairs]
    references = [x[1] for x in tokenized_pairs]
    bleu = sacrebleu.corpus_bleu(hypotheses=hypotheses, references=[references], tokenize="none")
    return {
        "bleu": float(bleu.score),
        "bp": float(bleu.bp),
        "sys_len": int(bleu.sys_len),
        "ref_len": int(bleu.ref_len),
        "precisions": [float(x) for x in bleu.precisions],
        "used_samples": len(tokenized_pairs),
        "skipped_empty_after_tokenize": skipped_empty,
        "tokenizer": "mecab" if use_mecab else "none",
    }


def parse_torch_dtype(value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"auto", ""}:
        return "auto"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch が必要です。") from exc

    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --judge-torch-dtype: {value}")


def load_with_optional_token(from_pretrained: Any, model_name: str, kwargs: dict[str, Any]) -> Any:
    try:
        return from_pretrained(model_name, **kwargs)
    except TypeError:
        if "token" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["use_auth_token"] = retry_kwargs.pop("token")
        return from_pretrained(model_name, **retry_kwargs)


def load_qwen_judge_model(
    model_name: str,
    device_map: str,
    torch_dtype: Any,
    attn_implementation: Optional[str],
) -> tuple[Any, Any, str]:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "transformers が必要です。`pip install transformers` を実行してください。"
        ) from exc

    hf_token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    hf_token = hf_token.strip() if isinstance(hf_token, str) and hf_token.strip() else None

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        model_kwargs["token"] = hf_token
    if device_map and device_map.lower() != "none":
        model_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    loader_candidates: List[Tuple[str, Any, str]] = []
    qwen_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if qwen_cls is not None:
        loader_candidates.append(
            ("Qwen3VLForConditionalGeneration", qwen_cls.from_pretrained, "processor")
        )
    auto_img_text = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_img_text is not None:
        loader_candidates.append(
            ("AutoModelForImageTextToText", auto_img_text.from_pretrained, "processor")
        )
    auto_causal = getattr(transformers, "AutoModelForCausalLM", None)
    if auto_causal is not None:
        loader_candidates.append(
            ("AutoModelForCausalLM", auto_causal.from_pretrained, "tokenizer")
        )

    if not loader_candidates:
        raise RuntimeError("No usable model loader found in transformers.")

    model = None
    io_mode = ""
    errors: List[str] = []
    for loader_name, loader, candidate_mode in loader_candidates:
        try:
            model = load_with_optional_token(loader, model_name, model_kwargs)
            io_mode = candidate_mode
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader_name}: {exc}")

    if model is None:
        error_summary = " | ".join(errors)
        raise RuntimeError(f"Model load failed: {error_summary}")

    if io_mode == "processor":
        auto_processor = getattr(transformers, "AutoProcessor", None)
        if auto_processor is None:
            raise RuntimeError("transformers.AutoProcessor が見つかりません。")
        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            processor_kwargs["token"] = hf_token
        try:
            io_processor = load_with_optional_token(
                auto_processor.from_pretrained,
                model_name,
                processor_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            # Workaround for transformers versions where AutoProcessor fails
            # for some VL models (e.g., video processor class resolution).
            print(
                "[WARN] AutoProcessor load failed. Fallback to AutoTokenizer mode: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            io_mode = "tokenizer"
            tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
            io_processor = load_with_optional_token(
                transformers.AutoTokenizer.from_pretrained,
                model_name,
                tokenizer_kwargs,
            )
            if (
                getattr(io_processor, "pad_token", None) is None
                and getattr(io_processor, "eos_token", None) is not None
            ):
                io_processor.pad_token = io_processor.eos_token
    else:
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
        io_processor = load_with_optional_token(
            transformers.AutoTokenizer.from_pretrained,
            model_name,
            tokenizer_kwargs,
        )
        if (
            getattr(io_processor, "pad_token", None) is None
            and getattr(io_processor, "eos_token", None) is not None
        ):
            io_processor.pad_token = io_processor.eos_token

    model.eval()
    return model, io_processor, io_mode


def resolve_input_device(model: Any) -> Optional[Any]:
    try:
        import torch
    except ImportError:
        return None

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for _module_name, mapped_device in hf_device_map.items():
            if mapped_device in {None, "cpu", "disk"}:
                continue
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str):
                try:
                    return torch.device(mapped_device)
                except Exception:  # noqa: BLE001
                    continue

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def generate_judge_response(
    model: Any,
    io_processor: Any,
    io_mode: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch が必要です。") from exc

    if hasattr(io_processor, "apply_chat_template"):
        if io_mode == "processor":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": prompt}]
        text_prompt = io_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text_prompt = prompt

    if io_mode == "processor":
        try:
            inputs = io_processor(text=[text_prompt], padding=True, return_tensors="pt")
        except TypeError:
            inputs = io_processor(text=[text_prompt], return_tensors="pt")
    else:
        inputs = io_processor(text_prompt, return_tensors="pt")

    input_device = resolve_input_device(model)
    if input_device is not None:
        inputs = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0 or top_p < 1.0 or top_k > 0:
        gen_kwargs["do_sample"] = True
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = 0
    input_ids = inputs.get("input_ids")
    if hasattr(input_ids, "shape") and len(input_ids.shape) == 2:
        prompt_len = int(input_ids.shape[-1])

    if prompt_len > 0 and hasattr(generated_ids, "shape") and generated_ids.shape[-1] > prompt_len:
        generated_ids = generated_ids[:, prompt_len:]

    decoded = io_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def _extract_score_from_json_text(text: str) -> Optional[int]:
    match = JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    score = payload.get("score")
    if isinstance(score, int) and 1 <= score <= 5:
        return score
    if isinstance(score, str) and score.strip() in {"1", "2", "3", "4", "5"}:
        return int(score.strip())
    return None


def parse_score(raw_text: str) -> Optional[int]:
    cleaned = raw_text.strip().translate(FULLWIDTH_TO_ASCII)
    if not cleaned:
        return None

    score = _extract_score_from_json_text(cleaned)
    if score is not None:
        return score

    if cleaned in {"1", "2", "3", "4", "5"}:
        return int(cleaned)

    digit_match = re.search(r"(?:^|[^0-9])([1-5])(?:[^0-9]|$)", cleaned)
    if digit_match:
        return int(digit_match.group(1))

    lowered = cleaned.lower()
    for phrase, value in LABEL_SCORE_RULES:
        if phrase in lowered:
            return value
    for word, value in ENGLISH_SCORE_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def run_llm_judge(
    samples: Sequence[EvalSample],
    model_name: str,
    judge_max_samples: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device_map: str,
    torch_dtype_text: str,
    attn_implementation: Optional[str],
    fallback_score_on_parse_failure: int,
) -> dict[str, Any]:
    judge_samples = list(samples[:judge_max_samples] if judge_max_samples is not None else samples)
    if not judge_samples:
        return {
            "judge_model": model_name,
            "used_samples": 0,
            "mean_score": 0.0,
            "fallback_score_used": 0,
            "raw_parse_failures": 0,
            "scores": [],
        }

    torch_dtype = parse_torch_dtype(torch_dtype_text)
    model, io_processor, io_mode = load_qwen_judge_model(
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    scores: List[int] = []
    raw_parse_failures = 0
    fallback_used = 0
    per_sample_rows: List[dict[str, Any]] = []

    report_every = max(1, len(judge_samples) // 20)
    for idx, sample in enumerate(judge_samples, start=1):
        prompt = DEFAULT_JUDGE_TEMPLATE.format(
            Question=sample.question,
            Answer=sample.reference,
            Prediction=sample.prediction,
        )
        raw_response = generate_judge_response(
            model=model,
            io_processor=io_processor,
            io_mode=io_mode,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        parsed_score = parse_score(raw_response)
        if parsed_score is None:
            raw_parse_failures += 1
            parsed_score = fallback_score_on_parse_failure
            fallback_used += 1
        scores.append(parsed_score)

        per_sample_rows.append(
            {
                "sample_id": sample.sample_id,
                "score": parsed_score,
                "raw_response": raw_response,
            }
        )

        if idx % report_every == 0 or idx == len(judge_samples):
            print(
                f"[Judge Progress] {idx}/{len(judge_samples)}",
                file=sys.stderr,
            )

    mean_score = statistics.mean(scores) if scores else 0.0
    return {
        "judge_model": model_name,
        "used_samples": len(scores),
        "mean_score": float(mean_score),
        "fallback_score_used": fallback_used,
        "raw_parse_failures": raw_parse_failures,
        "scores": scores,
        "per_sample": per_sample_rows,
    }


def main() -> int:
    args = parse_args()

    if not args.predictions_jsonl.exists():
        print(f"[ERROR] predictions JSONL not found: {args.predictions_jsonl}", file=sys.stderr)
        return 1
    if not args.qa_jsonl.exists():
        print(f"[ERROR] QA JSONL not found: {args.qa_jsonl}", file=sys.stderr)
        return 1
    if not (1 <= args.fallback_score_on_parse_failure <= 5):
        print("[ERROR] --fallback-score-on-parse-failure は 1..5 を指定してください。", file=sys.stderr)
        return 1
    if args.judge_max_samples is not None and args.judge_max_samples <= 0:
        print("[ERROR] --judge-max-samples は正の整数を指定してください。", file=sys.stderr)
        return 1
    if args.judge_max_new_tokens <= 0:
        print("[ERROR] --judge-max-new-tokens は正の整数を指定してください。", file=sys.stderr)
        return 1
    if not (0.0 < args.judge_top_p <= 1.0):
        print("[ERROR] --judge-top-p は (0, 1] を指定してください。", file=sys.stderr)
        return 1
    if args.judge_top_k < 0:
        print("[ERROR] --judge-top-k は 0 以上を指定してください。", file=sys.stderr)
        return 1

    answer_keys = read_answer_keys(args.answer_keys)
    prediction_rows = load_jsonl(args.predictions_jsonl)
    qa_rows = load_jsonl(args.qa_jsonl)

    samples, pairing_stats = build_eval_samples(
        prediction_rows=prediction_rows,
        qa_rows=qa_rows,
        prediction_key=args.prediction_key,
        question_key=args.question_key,
        answer_keys=answer_keys,
        include_non_ok=args.include_non_ok,
        match_key=args.match_key,
        pred_match_key=args.pred_match_key,
        qa_match_key=args.qa_match_key,
    )

    if not samples:
        print("[ERROR] 評価対象サンプルがありません。キー指定やJSONLの整合を確認してください。", file=sys.stderr)
        return 1

    bleu_result = compute_corpus_bleu(
        samples=samples,
        use_mecab=not args.disable_mecab,
        mecab_args=args.mecab_args,
    )

    judge_result = run_llm_judge(
        samples=samples,
        model_name=args.judge_model,
        judge_max_samples=args.judge_max_samples,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        top_k=args.judge_top_k,
        device_map=args.judge_device_map,
        torch_dtype_text=args.judge_torch_dtype,
        attn_implementation=args.judge_attn_implementation,
        fallback_score_on_parse_failure=args.fallback_score_on_parse_failure,
    )

    print("[Pairing]")
    print(f"mode: {pairing_stats.pairing_mode}")
    print(f"used_samples: {pairing_stats.used_samples}")
    print(
        "skipped: "
        f"status={pairing_stats.skipped_status}, "
        f"missing_prediction={pairing_stats.skipped_missing_prediction}, "
        f"missing_question={pairing_stats.skipped_missing_question}, "
        f"missing_answer={pairing_stats.skipped_missing_answer}, "
        f"missing_match={pairing_stats.skipped_missing_match}"
    )

    print("\n[Corpus BLEU]")
    print(f"BLEU: {bleu_result['bleu']:.4f}")
    print(f"used_samples: {bleu_result['used_samples']}")
    print(f"tokenizer: {bleu_result['tokenizer']}")

    print("\n[LLM-as-a-judge]")
    print(f"judge_model: {judge_result['judge_model']}")
    print(f"mean_score: {judge_result['mean_score']:.4f}")
    print(f"used_samples: {judge_result['used_samples']}")
    print(f"fallback_score_used: {judge_result['fallback_score_used']}")

    payload = {
        "input": {
            "predictions_jsonl": str(args.predictions_jsonl),
            "qa_jsonl": str(args.qa_jsonl),
            "prediction_key": args.prediction_key,
            "question_key": args.question_key,
            "answer_keys": answer_keys,
            "match_key": args.match_key,
            "pred_match_key": args.pred_match_key,
            "qa_match_key": args.qa_match_key,
        },
        "pairing": asdict(pairing_stats),
        "corpus_bleu": bleu_result,
        "llm_judge": judge_result,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
