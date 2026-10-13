
import matplotlib.pyplot as plt
import seaborn as sns


# 1. CLASS DISTRIBUTION
def plot_class_distribution(y, class_names):

    plt.figure(figsize=(7, 5))

    sns.countplot(x=y, color="steelblue")

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(
        ticks=range(len(class_names)),
        labels=class_names
    )

    plt.tight_layout()


# 2. CONFUSION MATRIX
def plot_confusion_matrix(cm, class_names):

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()


# 3. ROC CURVES
def plot_roc_curves(roc_curves, class_names):

    plt.figure(figsize=(7, 5))

    for class_label, values in roc_curves.items():

        fpr, tpr, auc_score = values

        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[class_label]} (AUC = {auc_score:.3f})"
        )

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier"
    )

    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()


# 4. DISPLAY ALL GRAPHS
def show_plots():
    plt.show()
