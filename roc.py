
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

def calculate_roc(y_true, y_prob, classes):

    # Convert inputs into NumPy arrays
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    classes = np.asarray(classes)

    curves = {}

    # Binary classification
    if len(classes) == 2:

        positive_class = classes[1]

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob[:, 1],
            pos_label=positive_class
        )

        auc_score = roc_auc_score(
            (y_true == positive_class).astype(int),
            y_prob[:, 1]
        )

        curves[positive_class] = (fpr, tpr, auc_score)

    # Multiclass classification
    else:

        for i, class_label in enumerate(classes):

            # One-vs-Rest: current class = 1, others = 0
            binary_target = (y_true == class_label).astype(int)

            fpr, tpr, _ = roc_curve(
                binary_target,
                y_prob[:, i]
            )

            class_auc = auc(fpr, tpr)

            curves[class_label] = (fpr, tpr, class_auc)

        # Overall multiclass ROC-AUC
        auc_score = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=classes
        )

    return auc_score, curves
