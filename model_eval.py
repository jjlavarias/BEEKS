import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from roc import calculate_roc


def evaluate_model(X, y, model, parameters, class_names):

    # 1. K-FOLD CROSS VALIDATION
    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=11
    )

    classes = np.sort(np.unique(y))

    accuracy_scores = []
    auc_scores = []

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    # 2. TRAINING AND GRID SEARCH
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Use the correct ROC-AUC scoring method
        if len(classes) == 2:
            scoring = "roc_auc"
        else:
            scoring = "roc_auc_ovr"

        grid = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        # Best model from GridSearchCV
        best_model = grid.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # ROC-AUC
        auc_score, _ = calculate_roc(
            y_test, y_prob, best_model.classes_
        )

        accuracy_scores.append(accuracy)
        auc_scores.append(auc_score)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

        print(f"\nFOLD {fold}")
        print("Best Parameters:", grid.best_params_)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc_score:.4f}")

    # 3. FINAL EVALUATION
    mean_accuracy = np.mean(accuracy_scores)
    mean_auc = np.mean(auc_scores)

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=classes
    )

    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=classes,
        target_names=class_names,
        zero_division=0
    )

    print("\nAVERAGE RESULTS")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean ROC-AUC: {mean_auc:.4f}")

    print("\nCONFUSION MATRIX")
    print(cm)

    print("\nCLASSIFICATION REPORT")
    print(report)

    # ROC curves from out-of-fold predictions
    overall_auc, roc_curves = calculate_roc(
        y_true_all,
        np.array(y_prob_all),
        classes
    )

    return {
        "accuracy_scores": accuracy_scores,
        "auc_scores": auc_scores,
        "mean_accuracy": mean_accuracy,
        "mean_auc": mean_auc,
        "confusion_matrix": cm,
        "roc_curves": roc_curves,
        "overall_auc": overall_auc
    }
