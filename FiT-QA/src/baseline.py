#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run QA inference on JSONL and write prediction JSONL.

Input:
- QA JSONL (e.g., test.jsonl) that has `question` key by default.

Output:
- Prediction JSONL where each row contains at least:
  - `status` ("ok" or "error")
  - `prediction` (or user-specified prediction key)
  - original QA row fields (kept as-is)

This output format is compatible with evaluate_prediction_jsonl.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "あなたはQAアシスタントです。"
    "質問に対して、短い語句で簡潔に答えてください。"
    "説明は不要です。"
)
JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
ANSWER_JSON_KEYS = ("answer", "prediction", "output", "response", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QA JSONL を読み、Hugging Face モデルで推論して prediction JSONL を出力します。"
        )
    )
    parser.add_argument(
        "--qa-jsonl",
        type=Path,
        required=True,
        help="question を含む JSONL (例: test.jsonl)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="推論結果 JSONL の出力先",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="質問テキストのキー (default: question)",
    )
    parser.add_argument(
        "--prediction-key",
        type=str,
        default="prediction",
        help="出力時の予測テキストキー (default: prediction)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face のモデル名 (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="推論時の system prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="生成最大トークン数 (default: 64)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成 temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="生成 top_p (default: 1.0)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="transformers model の device_map (default: auto)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="auto / bf16 / fp16 / fp32",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="attn_implementation を明示する場合に指定",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="parse失敗/生成失敗時の最大試行回数 (default: 3)",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=0.0,
        help="リトライ間の待機秒 (default: 0.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭N件のみ実行",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="output-jsonl が存在する場合に上書きする",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse failed: {path}:{line_no}", file=sys.stderr)
                continue
            if not isinstance(row, dict):
                print(f"[WARN] Non-object JSON ignored: {path}:{line_no}", file=sys.stderr)
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def parse_torch_dtype(value: str) -> Any:
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
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
    raise ValueError(f"Unsupported --torch-dtype: {value}")


def load_with_optional_token(from_pretrained: Any, model_name: str, kwargs: Dict[str, Any]) -> Any:
    try:
        return from_pretrained(model_name, **kwargs)
    except TypeError:
        if "token" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["use_auth_token"] = retry_kwargs.pop("token")
        return from_pretrained(model_name, **retry_kwargs)


def load_hf_model(
    model_name: str,
    device_map: str,
    torch_dtype: Any,
    attn_implementation: Optional[str],
) -> tuple[Any, Any, str]:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("transformers が必要です。`pip install transformers` を実行してください。") from exc

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    hf_token = hf_token.strip() if isinstance(hf_token, str) and hf_token.strip() else None

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        model_kwargs["token"] = hf_token
    if device_map and device_map.lower() != "none":
        model_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    loader_candidates: List[tuple[str, Any, str]] = []
    qwen_vl_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if qwen_vl_cls is not None:
        loader_candidates.append(
            ("Qwen3VLForConditionalGeneration", qwen_vl_cls.from_pretrained, "processor")
        )
    auto_img_text = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_img_text is not None:
        loader_candidates.append(
            ("AutoModelForImageTextToText", auto_img_text.from_pretrained, "processor")
        )
    auto_causal = getattr(transformers, "AutoModelForCausalLM", None)
    if auto_causal is not None:
        loader_candidates.append(("AutoModelForCausalLM", auto_causal.from_pretrained, "tokenizer"))

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
        raise RuntimeError("Model load failed: " + " | ".join(errors))

    if io_mode == "processor":
        auto_processor = getattr(transformers, "AutoProcessor", None)
        if auto_processor is None:
            raise RuntimeError("transformers.AutoProcessor が見つかりません。")
        processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            processor_kwargs["token"] = hf_token
        try:
            io_processor = load_with_optional_token(
                auto_processor.from_pretrained,
                model_name,
                processor_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            # Some transformers versions fail to build AutoProcessor for specific
            # VL models (e.g., video processor class resolution). For text-only
            # QA, fallback to tokenizer mode to keep inference running.
            print(
                "[WARN] AutoProcessor load failed. Fallback to AutoTokenizer mode: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            io_mode = "tokenizer"
            tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
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
        tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
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


def build_chat_prompt(chat_template_owner: Any, system_prompt: str, question: str) -> str:
    if hasattr(chat_template_owner, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return chat_template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"System:\n{system_prompt}\n\n"
        f"User:\n{question}\n\n"
        "Assistant:\n"
    )


def generate_answer(
    model: Any,
    io_processor: Any,
    io_mode: str,
    system_prompt: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch が必要です。") from exc

    text_prompt = build_chat_prompt(io_processor, system_prompt=system_prompt, question=question)
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

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0 or top_p < 1.0:
        gen_kwargs["do_sample"] = True
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
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


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    match = JSON_OBJECT_PATTERN.search(stripped)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _cleanup_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    normalized_lines: List[str] = []
    for line in lines:
        line = re.sub(r"^(assistant|回答|answer|prediction)\s*[:：]\s*", "", line, flags=re.IGNORECASE)
        if line:
            normalized_lines.append(line)

    if normalized_lines:
        cleaned = normalized_lines[0]

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()

    return cleaned


def parse_prediction_text(raw_text: str) -> Optional[str]:
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None

    payload = _extract_json_object(text)
    if payload is not None:
        for key in ANSWER_JSON_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            parsed = _cleanup_text(str(value))
            if parsed:
                return parsed
        for value in payload.values():
            if isinstance(value, (str, int, float, bool)):
                parsed = _cleanup_text(str(value))
                if parsed:
                    return parsed

    parsed = _cleanup_text(text)
    return parsed or None


def build_record(
    qa_row: Dict[str, Any],
    line_no: int,
    question_key: str,
    prediction_key: str,
    model_name: str,
    prediction: str,
    raw_prediction: str,
    attempts: int,
    parse_retry_count: int,
    error_message: Optional[str],
    latency_sec: float,
) -> Dict[str, Any]:
    record = {k: v for k, v in qa_row.items() if not str(k).startswith("_")}
    record["sample_id"] = (
        str(record.get("sample_id", "")).strip()
        or str(record.get("question_id", "")).strip()
        or f"line:{line_no}"
    )
    record["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["status"] = "ok" if error_message is None else "error"
    record["error"] = error_message
    record["latency_sec"] = round(latency_sec, 4)
    record["model_name"] = model_name
    record["qa_line"] = line_no
    record["question_key"] = question_key
    record["inference_attempts"] = attempts
    record["parse_retry_count"] = parse_retry_count
    record["raw_prediction"] = raw_prediction
    record[prediction_key] = prediction
    return record


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    args = parse_args()

    if not args.qa_jsonl.exists():
        print(f"[ERROR] QA JSONL not found: {args.qa_jsonl}", file=sys.stderr)
        return 1
    if args.max_new_tokens <= 0:
        print("[ERROR] --max-new-tokens は正の整数を指定してください。", file=sys.stderr)
        return 1
    if not (0.0 < args.top_p <= 1.0):
        print("[ERROR] --top-p は (0, 1] を指定してください。", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit <= 0:
        print("[ERROR] --limit は正の整数を指定してください。", file=sys.stderr)
        return 1
    if args.max_attempts <= 0:
        print("[ERROR] --max-attempts は正の整数を指定してください。", file=sys.stderr)
        return 1
    if args.retry_sleep_seconds < 0:
        print("[ERROR] --retry-sleep-seconds は 0 以上を指定してください。", file=sys.stderr)
        return 1

    if args.output_jsonl.exists():
        if args.overwrite:
            args.output_jsonl.unlink()
        else:
            print(
                f"[ERROR] output already exists: {args.output_jsonl} "
                "(上書きする場合は --overwrite)",
                file=sys.stderr,
            )
            return 1

    qa_rows = load_jsonl(args.qa_jsonl)
    if args.limit is not None:
        qa_rows = qa_rows[: args.limit]

    if not qa_rows:
        print("[ERROR] QA JSONL から有効な行が読み込めませんでした。", file=sys.stderr)
        return 1

    torch_dtype = parse_torch_dtype(args.torch_dtype)
    model, io_processor, io_mode = load_hf_model(
        model_name=args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    report_every = max(1, len(qa_rows) // 20)

    for idx, row in enumerate(qa_rows, start=1):
        line_no = int(row.get("_line_no", -1))
        question_raw = row.get(args.question_key)
        error_message: Optional[str] = None
        prediction = ""
        raw_prediction = ""
        attempts = 0
        parse_retry_count = 0

        start = time.time()
        if question_raw is None or not str(question_raw).strip():
            error_message = f"missing_question_key:{args.question_key}"
        else:
            question_text = str(question_raw).strip()
            last_exception_message: Optional[str] = None
            parse_failed_any = False

            for attempt in range(1, args.max_attempts + 1):
                attempts = attempt
                try:
                    raw_prediction = generate_answer(
                        model=model,
                        io_processor=io_processor,
                        io_mode=io_mode,
                        system_prompt=args.system_prompt,
                        question=question_text,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exception_message = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_attempts and args.retry_sleep_seconds > 0:
                        time.sleep(args.retry_sleep_seconds)
                    continue

                parsed_prediction = parse_prediction_text(raw_prediction)
                if parsed_prediction is not None:
                    prediction = parsed_prediction
                    error_message = None
                    break

                parse_retry_count += 1
                parse_failed_any = True
                if attempt < args.max_attempts and args.retry_sleep_seconds > 0:
                    time.sleep(args.retry_sleep_seconds)

            if not prediction:
                if parse_failed_any and last_exception_message is None:
                    error_message = f"parse_failed_after_{attempts}_attempts"
                elif parse_failed_any and last_exception_message is not None:
                    error_message = (
                        f"parse_failed_and_generation_error_after_{attempts}_attempts: "
                        f"{last_exception_message}"
                    )
                elif last_exception_message is not None:
                    error_message = f"{last_exception_message} (after {attempts} attempts)"
                else:
                    error_message = f"inference_failed_after_{attempts}_attempts"
        latency_sec = time.time() - start

        record = build_record(
            qa_row=row,
            line_no=line_no,
            question_key=args.question_key,
            prediction_key=args.prediction_key,
            model_name=args.model,
            prediction=prediction,
            raw_prediction=raw_prediction,
            attempts=attempts,
            parse_retry_count=parse_retry_count,
            error_message=error_message,
            latency_sec=latency_sec,
        )
        append_jsonl(args.output_jsonl, [record])

        if error_message is None:
            success += 1
        else:
            failed += 1

        if idx % report_every == 0 or idx == len(qa_rows):
            print(
                f"[Inference Progress] {idx}/{len(qa_rows)} "
                f"(ok={success}, error={failed})",
                file=sys.stderr,
            )

    print("[Done]")
    print(f"model: {args.model}")
    print(f"input_rows: {len(qa_rows)}")
    print(f"ok: {success}")
    print(f"error: {failed}")
    print(f"saved: {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
