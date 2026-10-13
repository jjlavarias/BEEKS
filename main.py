
from data_loader import load_data
from logistic_regression import (
    create_model,
    get_parameters
)
from model_eval import evaluate_model
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    show_plots
)

def main():

    # 1. LOAD DATA
    X, y, class_names = load_data()

    # 2. CREATE LOGISTIC REGRESSION
    model = create_model()
    parameters = get_parameters()

    # 3. EVALUATE MODEL
    results = evaluate_model(
        X,
        y,
        model,
        parameters,
        class_names
    )

    # 4. DISPLAY GRAPHS
    plot_class_distribution(y, class_names)
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names
    )
    plot_roc_curves(
        results["roc_curves"],
        class_names
    )
    show_plots()

if __name__ == "__main__":
    main()
