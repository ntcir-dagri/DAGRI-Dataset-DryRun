from dagri_subtask1_eval.usecase.evaluate_usecase import EvaluateUsecase
from dagri_subtask1_eval.main.container import build_evaluate_usecase


def test_build_evaluate_usecase_returns_usecase() -> None:
    usecase = build_evaluate_usecase()

    assert isinstance(usecase, EvaluateUsecase)
