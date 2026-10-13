# Resources
# https://www.youtube.com/watch?v=aL21Y-u0SRs  : yt vid this code is based off of
# https://www.youtube.com/watch?v=HYcXgN9HaTM  : extra resource for comprehension
# https://www.youtube.com/watch?v=4u81xU7BIOc  : Sturms vid

# IMPORTS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# 1. LOAD IRIS DATASET
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target  # 0: setosa, 1: versicolor, 2: virginica

# 2. SELECT FEATURES AND TARGET
x = df[['sepal length (cm)', 'sepal width (cm)']]
y = df['species']

# 3. K-FOLD CROSS-VALIDATION
# Using StratifiedKFold to maintain equal class proportions across all 5 folds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
model = LogisticRegression(max_iter=200)

print("=== K-FOLD CROSS-VALIDATION RESULTS ===")

# Method A: Quick summary of scores across all 5 folds
scores = cross_val_score(model, x, y, cv=kf)
print(f"Accuracy scores per fold: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})\n")

# Method B: Iterating through folds to compute overall confusion matrix
y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(kf.split(x, y), 1):
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    # Store fold results to aggregate metrics across the entire dataset
    y_true_all.extend(y_test)
    y_pred_all.extend(y_predict)

# 4. AGGREGATE EVALUATION METRICS
print("=== AGGREGATED CONFUSION MATRIX (All Folds) ===")
print(confusion_matrix(y_true_all, y_pred_all))

print("\n=== AGGREGATED CLASSIFICATION REPORT ===")
print(classification_report(y_true_all, y_pred_all, target_names=iris.target_names))

# 5. DISPLAY PLOTS
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
plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names))

plt.figure(figsize=(6, 4))
sns.countplot(x='species', data=df)
plt.title('Count of Iris Species Classes')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)

plt.show()