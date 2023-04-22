import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix

# Load the dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('data/bezdekIris.data', names=column_names)

# Encode class labels as integers
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

# Split the dataset into features (X) and labels (y)
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Hyperparameter tuning using GridSearchCV
knn_params = {'n_neighbors': list(range(1, 31))}
dt_params = {'max_depth': list(range(1, 11))}
svm_params = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}

knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
svm = GridSearchCV(SVC(random_state=42), svm_params, cv=5)

# Train the models
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Evaluate the models on the validation set
knn_val_preds = knn.predict(X_val)
dt_val_preds = dt.predict(X_val)
svm_val_preds = svm.predict(X_val)

knn_val_acc = accuracy_score(y_val, knn_val_preds)
dt_val_acc = accuracy_score(y_val, dt_val_preds)
svm_val_acc = accuracy_score(y_val, svm_val_preds)

print(f"k-NN Validation Accuracy: {knn_val_acc}")
print(f"Decision Tree Validation Accuracy: {dt_val_acc}")
print(f"SVM Validation Accuracy: {svm_val_acc}")

# Perform 5-fold cross-validation for each model
knn_cv_scores = cross_val_score(KNeighborsClassifier(**knn.best_params_), X, y, cv=5)
dt_cv_scores = cross_val_score(DecisionTreeClassifier(**dt.best_params_, random_state=42), X, y, cv=5)
svm_cv_scores = cross_val_score(SVC(**svm.best_params_, random_state=42), X, y, cv=5)

# Calculate the mean cross-validation accuracy for each model
knn_cv_mean_acc = knn_cv_scores.mean()
dt_cv_mean_acc = dt_cv_scores.mean()
svm_cv_mean_acc = svm_cv_scores.mean()

print(f"k-NN 5-Fold Cross-Validation Mean Accuracy: {knn_cv_mean_acc}")
print(f"Decision Tree 5-Fold Cross-Validation Mean Accuracy: {dt_cv_mean_acc}")
print(f"SVM 5-Fold Cross-Validation Mean Accuracy: {svm_cv_mean_acc}")


# Calculate metrics for each model using test set
for model, name in [(knn, "k-NN"), (dt, "Decision Tree"), (svm, "SVM")]:
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')

    print(f"{name} Test Precision: {precision}")
    print(f"{name} Test Recall: {recall}")
    print(f"{name} Test F1-score: {f1}")
    print()

    # Plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# Function to plot decision boundaries for 2D data
def plot_decision_boundary(model, X, y, ax, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Classes")

# Select two features for visualization
X_train_2d = X_train.iloc[:, :2].values
X_test_2d = X_test.iloc[:, :2].values

# Train models with 2D data
knn_2d = KNeighborsClassifier(**knn.best_params_).fit(X_train_2d, y_train)
dt_2d = DecisionTreeClassifier(**dt.best_params_, random_state=42).fit(X_train_2d, y_train)
svm_2d = SVC(**svm.best_params_, random_state=42).fit(X_train_2d, y_train)

# Plot decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_decision_boundary(knn_2d, X_test_2d, y_test.values, axes[0], "k-NN Decision Boundary")
plot_decision_boundary(dt_2d, X_test_2d, y_test.values, axes[1], "Decision Tree Decision Boundary")
plot_decision_boundary(svm_2d, X_test_2d, y_test.values, axes[2], "SVM Decision Boundary")

plt.show()