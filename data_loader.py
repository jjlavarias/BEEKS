from sklearn.datasets import load_iris

def load_data():

    # Load Iris dataset
    iris = load_iris(as_frame=True)

    # X = measurements used for predictions
    X = iris.data

    # y = correct classification labels
    y = iris.target

    # Names of the classifications
    class_names = iris.target_names

    return X, y, class_names
