# Resources:
# https://www.youtube.com/watch?v=aL21Y-u0SRs
# https://www.youtube.com/watch?v=HYcXgN9HaTM
# https://www.youtube.com/watch?v=4u81xU7BIOc

from data_loader import load_iris_data, prepare_features_and_target
from model_eval import evaluate_kfold, print_evaluation_reports
from visualization import plot_feature_scatter, plot_class_distribution, show_plots

def main():
    # 1. Load Data
    df, target_names = load_iris_data()
    x, y = prepare_features_and_target(df)

    # 2. Evaluate Model
    scores, y_true, y_pred = evaluate_kfold(x, y, n_splits=5, random_state=11)
    print_evaluation_reports(scores, y_true, y_pred, target_names)

    # 3. Render Visualizations
    plot_feature_scatter(df, target_names)
    plot_class_distribution(df, target_names)
    show_plots()

if __name__ == "__main__":
    main()