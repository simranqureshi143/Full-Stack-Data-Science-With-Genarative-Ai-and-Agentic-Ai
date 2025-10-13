import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load dataset
dataset = pd.read_csv(r"C:\Users\admin\Downloads\logit classification.csv")

# Step 2: Split into features (X) and target (Y)
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, -1].values

# Step 3: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Step 4: Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5: SVM model training
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)   # You can also try 'rbf', 'poly', or 'sigmoid'
classifier.fit(X_train, Y_train)

# Step 6: Predictions
Y_pred = classifier.predict(X_test)

# Step 7: Model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)

ac = accuracy_score(Y_test, Y_pred)
print("\nAccuracy:", ac)

cr = classification_report(Y_test, Y_pred)
print("\nClassification Report:\n", cr)

# Step 8: Bias and Variance
bias = classifier.score(X_train, Y_train)
print("\nTraining Accuracy (Bias):", bias)

variance = classifier.score(X_test, Y_test)
print("Testing Accuracy (Variance):", variance)

# Step 9: Model prediction on future/unseen data
dataset1 = pd.read_csv(r"C:\Users\admin\Downloads\logit classification.csv")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2, 3]].values
M = sc.transform(dataset1)

d2['Y_pred1'] = classifier.predict(M)

d2.to_csv('final.csv', index=False)

# Step 10: Check where the file is saved
import os
print("\nFile saved at:", os.getcwd())

