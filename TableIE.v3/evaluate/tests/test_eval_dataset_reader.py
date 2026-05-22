from pathlib import Path

import pytest

from dagri_subtask1_eval.infra.dataset_reader_error import DatasetValidationError
from dagri_subtask1_eval.infra.eval_dataset_reader import EvalDatasetReader


def test_read_returns_eval_dataset_from_jsonl() -> None:
    reader = EvalDatasetReader()

    dataset = reader.read(Path("examples/test_groundtruth.jsonl"))

    assert len(dataset.items) == 7

    first_item = dataset.items[0]
    assert first_item.prefecture_name == "nagasaki"
    assert first_item.id == "1727423373"
    assert len(first_item.management_types) == 1
    assert first_item.management_types[0].id == "onion"
    assert first_item.management_types[0].growing_area.items[0].area == 100
    assert len(first_item.management_indicators) == 1
    assert first_item.management_indicators[0].crop_name == "たまねぎ（加工・業務用）"
    assert first_item.management_indicators[0].work_schedule.items[0].period.value == "1月上旬"


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '\n{"prefecture_name":"tokyo","id":"1","management_types":[],"management_indicators":[]}\n\n',
        encoding="utf-8",
    )

    reader = EvalDatasetReader()

    dataset = reader.read(dataset_file)

    assert len(dataset.items) == 1
    assert dataset.items[0].prefecture_name == "tokyo"


def test_read_collects_all_validation_issues_before_raising() -> None:
    reader = EvalDatasetReader()
    dataset_file = Path("tests/data_invalid_eval.jsonl")

    with pytest.raises(DatasetValidationError) as exc_info:
        reader.read(dataset_file)

    issues = exc_info.value.issues

    assert len(issues) == 3
    assert issues[0].line_number == 1
    assert issues[0].field_path == ("management_types",)
    assert issues[1].line_number == 1
    assert issues[1].field_path == ("management_indicators",)
    assert issues[2].line_number == 2
    assert issues[2].field_path == ("id",)
