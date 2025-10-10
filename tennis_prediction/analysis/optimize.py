import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd

df = pd.read_csv("../data/preprocessed/preprocessed_all.csv")
X = df.drop(columns=["player_id", "opponent_id", "winner"])
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

param_grid = {
    "n_estimators": [150, 200, 250, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None]
}

search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, n_iter=50, cv=5, scoring='accuracy', n_jobs=8, verbose=2, random_state=42)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best CV accuracy:", search.best_score_)

best_model = search.best_estimator_

probs = best_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0, 1, 1000)
accuracies = []

for t in thresholds:
    preds = (probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

best_index = np.argmax(accuracies)
best_threshold = thresholds[best_index]
best_accuracy = accuracies[best_index]

final_preds = (probs >= best_threshold).astype(int)

print(f"Best threshold for max accuracy: {best_threshold:.4f}")
print(f"Maximum accuracy: {best_accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, final_preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))
