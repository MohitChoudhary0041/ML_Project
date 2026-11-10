
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# 1. Load Dataset
df = pd.read_csv("student-mat.csv", sep=";")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:", df.shape)


# 2. Data Cleaning and Inspection

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nFinal Grade Distribution:")
print(df["G3"].value_counts().sort_index())


# 3. Categorize Student Performance

def categorize_grade(grade):
    if grade < 10:
        return "Low"
    elif grade < 15:
        return "Average"
    else:
        return "High"


df["Performance"] = df["G3"].apply(categorize_grade)

print("\nStudent Performance Categories:")
print(df["Performance"].value_counts())


# 4. Separate Features and Target

X = df.drop(["G3", "Performance"], axis=1)
y = df["Performance"]

# Convert categorical data into numerical values
X = pd.get_dummies(X, drop_first=True)

print("\nProcessed Dataset:")
print(X.head())

print("\nTotal Features:", X.shape[1])


# 5. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# 6. Train Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")


# 7. Model Prediction

y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Performance")
plt.ylabel("Actual Performance")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()


# 9. Multiclass ROC Curve

# Convert actual labels into binary format
classes = model.classes_

y_test_binary = label_binarize(
    y_test,
    classes=classes
)

# Get prediction probabilities
y_prob = model.predict_proba(X_test)

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(classes):

    fpr, tpr, thresholds = roc_curve(
        y_test_binary[:, i],
        y_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {roc_auc:.2f})"
    )

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("Multiclass ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")
plt.grid()

plt.tight_layout()

plt.savefig("roc_curve.png")
plt.show()


print("\nROC Curve Generated Successfully!")
print("Project Evaluation Completed!")
