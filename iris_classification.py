# ============================================================
# IRIS FLOWER CLASSIFICATION (Beginner Friendly Version)
#BY SIDDHARTH GUPTA
# PCE23AD053
# ------------------------------------------------------------
# This program performs:
# 1. Exploratory Data Analysis (EDA)
# 2. Classification using Decision Tree, SVM, and KNN
# ============================================================

# Importing necessary libraries
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# Step 1: Load the IRIS dataset
# ------------------------------------------------------------
iris = datasets.load_iris()

# Create a DataFrame for easy handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\nFirst 5 rows of the dataset:")
print(df.head())

# ------------------------------------------------------------
# Step 2: Basic Information
# ------------------------------------------------------------
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nNumber of samples for each species:")
print(df['species'].value_counts())

# ------------------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# ------------------------------------------------------------

# (a) Distribution of features by species
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# (b) Correlation between features
plt.figure(figsize=(6, 5))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# (c) Boxplot to check for outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient='h')
plt.title("Boxplot of Features (Checking Outliers)")
plt.show()

# ------------------------------------------------------------
# Step 4: Prepare data for training
# ------------------------------------------------------------
X = df.iloc[:, :-1]   # feature columns
y = df['species']     # target column

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data (needed for SVM and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# Step 5: Train three supervised models
# ------------------------------------------------------------

# 1. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# 2. Support Vector Machine
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# 3. K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

# ------------------------------------------------------------
# Step 6: Evaluate model performance
# ------------------------------------------------------------

def evaluate_model(name, y_test, y_pred):
    print(f"\n====== {name} ======")
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate each model
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("Support Vector Machine (SVM)", y_test, svm_pred)
evaluate_model("K-Nearest Neighbors (KNN)", y_test, knn_pred)

# ------------------------------------------------------------
# Step 7: Compare accuracy
# ------------------------------------------------------------
results = {
    'Decision Tree': accuracy_score(y_test, dt_pred),
    'SVM': accuracy_score(y_test, svm_pred),
    'KNN': accuracy_score(y_test, knn_pred)
}

plt.bar(results.keys(), results.values(), color=['lightgreen', 'skyblue', 'salmon'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.0)
plt.show()

# ------------------------------------------------------------
# Step 8: Conclusion
# ------------------------------------------------------------
print("\n=== Conclusion ===")
print("→ All three models give high accuracy (above 90%).")
print("→ Decision Tree and SVM perform best for this dataset.")
print("→ Petal length and width are the most important features.")
