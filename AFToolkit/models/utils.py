import numpy as np  

from scipy import stats
from sklearn import metrics


def get_model_suffix(model, multi_train=False):
    model_suffix = ""
    if model.concat_features:
        model_suffix += "_concat"
    else:
        model_suffix += "_diff"
    if multi_train:
        model_suffix += "_multitrain"
    else:
        model_suffix += "_nomultitrain"
    model_suffix += "_agg" + model.protein_aggregation
    model_suffix += "_multi" + model.multi_aggregation
    return model_suffix


def calculate_metrics(predictions, targets, stab_threshold=-0.5):
    """Calculate metrics for ddG predictions."""
    # treat stabilizing mutations as positive class
    classification_targets = (targets < stab_threshold).astype(int)
    classification_predictions = (predictions < stab_threshold).astype(int)

    precision, recall, thresholds = metrics.precision_recall_curve(
        classification_targets, -predictions
    )

    stabilizing_muts_idx = np.where(targets < stab_threshold)[0]
    stabilizing_targets = targets[stabilizing_muts_idx]
    stabilizing_predictions = predictions[stabilizing_muts_idx]

    return {
        "PearsonR": stats.pearsonr(predictions, targets).statistic,
        "SpearmanR": stats.spearmanr(predictions, targets).statistic,
        "StabilizingPearsonR": stats.pearsonr(
            stabilizing_predictions, stabilizing_targets
        ).statistic,
        "StabilizingSpearmanR": stats.spearmanr(
            stabilizing_predictions, stabilizing_targets
        ).statistic,
        "RMSE": ((predictions - targets) ** 2).mean() ** 0.5,
        "ROC AUC": metrics.roc_auc_score(classification_targets, -predictions),
        "PRC AUC": metrics.auc(recall, precision),
        "AP": metrics.average_precision_score(classification_targets, -predictions),
        "MCC": metrics.matthews_corrcoef(
            classification_targets, classification_predictions
        ),
        "ACC": (classification_targets == classification_predictions).mean(),
        "R2": metrics.r2_score(targets, predictions),
        "Precision": metrics.precision_score(
            classification_targets,
            classification_predictions,
        ),
        "Recall": metrics.recall_score(
            classification_targets,
            classification_predictions,
        )
    }