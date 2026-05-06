from dagri_subtask1_eval.infra import (
    DefaultBalanceEvaluationService,
    DefaultCapitalEquipmentEvaluationService,
    DefaultGrowingAreaEvaluationService,
    DefaultManagementIndicatorBalanceEvaluationService,
    DefaultPremiseEvaluationService,
    DefaultWorkScheduleEvaluationService,
    DefaultWorkTechnologyEvaluationService,
    EvalDatasetReader,
    SimilarityBasedManagementIndicatorAlignmentService,
    SimilarityBasedManagementTypeAlignmentService,
    SubmissionDatasetReader,
)
from dagri_subtask1_eval.usecase.evaluate_usecase import EvaluateUsecase


def build_evaluate_usecase() -> EvaluateUsecase:
    return EvaluateUsecase(
        submission_dataset_reader=SubmissionDatasetReader(),
        eval_dataset_reader=EvalDatasetReader(),
        management_type_alignment_service=SimilarityBasedManagementTypeAlignmentService(),
        management_indicator_alignment_service=SimilarityBasedManagementIndicatorAlignmentService(),
        premise_evaluation_service=DefaultPremiseEvaluationService(),
        management_type_balance_evaluation_service=DefaultBalanceEvaluationService(),
        growing_area_evaluation_service=DefaultGrowingAreaEvaluationService(),
        capital_equipment_evaluation_service=DefaultCapitalEquipmentEvaluationService(),
        management_indicator_balance_evaluation_service=DefaultManagementIndicatorBalanceEvaluationService(),
        work_schedule_evaluation_service=DefaultWorkScheduleEvaluationService(),
        work_technology_evaluation_service=DefaultWorkTechnologyEvaluationService(),
    )
