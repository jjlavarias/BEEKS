from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_model():

    # Scale the data, then apply logistic regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000))
    ])

    return model


def get_parameters():

    # Values GridSearchCV will test
    parameters = {
        "logistic__C": [0.01, 0.1, 1, 10, 100],
        "logistic__class_weight": [None, "balanced"]
    }

    return parameters
