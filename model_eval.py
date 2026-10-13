import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_kfold(x, y, n_splits=5, random_state=11, max_iter=200):
    """Performs Stratified K-Fold cross-validation and collects aggregated predictions."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LogisticRegression(max_iter=max_iter)

    # Method A: Quick cross-validation scores
    scores = cross_val_score(model, x, y, cv=kf)

    # Method B: Iterating through folds for aggregated confusion matrix / report
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(x, y), 1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_predict)

    return scores, y_true_all, y_pred_all

def print_evaluation_reports(scores, y_true, y_pred, target_names):
    """Prints evaluation metrics including cross-validation scores, confusion matrix, and report."""
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"Accuracy scores per fold: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})\n")

    print("AGGREGATED CONFUSION MATRIX (All Folds)")
    print(confusion_matrix(y_true, y_pred))

    print("\nAGGREGATED CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=target_names))