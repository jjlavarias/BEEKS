from matplotlib import pyplot as plt
import seaborn as sns

def plot_feature_scatter(df, target_names):
    """Plots Sepal Length vs Sepal Width colored by Iris species."""
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        df['sepal length (cm)'],
        df['sepal width (cm)'],
        c=df['species'],
        cmap='viridis',
        edgecolor='k'
    )
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Sepal Length vs Sepal Width by Iris Species')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))

def plot_class_distribution(df, target_names):
    """Plots count distribution across Iris species classes."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='species', data=df)
    plt.title('Count of Iris Species Classes')
    plt.xticks(ticks=[0, 1, 2], labels=target_names)

def show_plots():
    """Renders all initialized plots."""
    plt.show()