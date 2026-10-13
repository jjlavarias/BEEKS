import pandas as pd
from sklearn.datasets import load_iris

def load_iris_data():
    """Loads raw Iris dataset and returns a clean DataFrame and target names."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

def prepare_features_and_target(df):
    """Extracts chosen feature matrix (X) and target vector (y)."""
    x = df[['sepal length (cm)', 'sepal width (cm)']]
    y = df['species']
    return x, y