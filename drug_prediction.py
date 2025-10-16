"""
LMNP
DS Practice
Drug Prescription Prediction
August 15th, 2025

Goal: building a machine learning that can predict
which drug a doctor should prescribe to a patient - based
on basic health features like age, sex, BP, cholesterol level,
Na-to-K (sodium to potassium ratio)

Plan:
1. Load data using pandas
2. Quick look on couples of first rows
3. Check for missing or incorrect data
4. Comprehend the columns
5. Convert text columns like Sex or BP into numbers using Label Encoding
6. Make sure the data is clean and ready to be train
7. Split data into training and test
8. Decision Tree classifier to train the patterns between patient features
and the prescribed drug
9. Check the accuracy, use accuracy, precision and recall metrics
10. Make decisions - input new patient data and aks the model to predict
which drug should be prescribed
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("drug200.csv")
df['Drug'] = df['Drug'].str.lower() # change all the drugs into lower case

# Quick look at the data
# print(df.head())
# print(df.info())

# create a label encoder
label = LabelEncoder()

# encode each categorical column
df['Sex'] = label.fit_transform(df['Sex']) # Female(F) = 0, Male(M) = 1
df['BP'] = label.fit_transform(df['BP']) # HIGH = 0, LOW = 1, NORMAL = 2
df['Cholesterol'] = label.fit_transform(df['Cholesterol']) # NORMAL = 0, HIGH = 1
df['Drug'] = label.fit_transform(df['Drug']) # DrugX = 0, DrugY = 1, etc.

print("Encoded drug classes:", list(label.classes_))

# Split the data in half, training and test
# training: used to train model
# testing: used to evaluate how well the model performs

# Separate features (X) and target (y)
X = df.drop("Drug", axis=1) # drop everything except the target
y = df["Drug"]                    # this is the target column

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Note: test_size=0.2 means 20% of the data will be used as the test set,
#       the remaining will be used for training the model
#       random_state=42 means setting the random seed used for splitting the data
#       Could be any number, but 42 is often used for convention

# Train a Decision Tree Classifier
# Create the model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the labels for test data
y_predict = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Classification Report:", classification_report(y_test, y_predict))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualization
plt.figure(figsize=(10, 10)) # this could be changed, ups to pref

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(cls) for cls in sorted(y.unique())],
    filled=True,
    rounded=True
)

plt.title("Decision Tree for Drug Prescription")
plt.savefig("drug.png")
plt.show()