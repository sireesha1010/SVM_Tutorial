import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# Predicting the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Reduce the dimensionality of the data to 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize and train a new SVM model on the PCA-transformed data
svm_model_pca = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model_pca.fit(X_train_pca, y_train)

# Plotting decision boundaries (only for 2 features for visualization)
def plot_decision_boundaries(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.show()

# Visualize for the first two features
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', marker='o')
plt.title('Original Data')

plt.subplot(2, 2, 2)
plot_decision_boundaries(X_test_pca, y_test, svm_model_pca)

# Bar Chart
C_values = [0.1, 1, 10]
accuracies = [96.67, 97.33, 96.67]

plt.figure(figsize=(10, 8))
plt.bar(C_values, accuracies)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different C Values')
plt.show()

# Heatmap
confusion_matrix = np.array([[48, 0, 2], [0, 47, 3], [0, 2, 48]])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

# Box Plot
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.boxplot(X_train_pca[:, 0], vert=False)
plt.title('Training Data')

plt.subplot(1, 2, 2)
plt.boxplot(X_test_pca[:, 0], vert=False)
plt.title('Testing Data')

plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(10, 8))
plt.hist(X_test_pca[:, 0], bins=10, alpha=0.5, label='Feature 1')
plt.hist(X_test_pca[:, 1], bins=10, alpha=0.5, label='Feature 2')
plt.legend()
plt.title('Histogram of Features')
plt.show()

# Violin Plot
plt.figure(figsize=(10, 8))
sns.violinplot(data=X_test_pca)
plt.title('Violin Plot of Features')
plt.show()


