from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

df = pd.read_csv("../data/preprocessed/preprocessed_all.csv")
X = df.drop(columns=["player_id", "opponent_id", "winner"])
y = df["winner"]

accuracies = []

for rs in range(0, 100):  # try 100 different splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=rs, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=186,
        random_state=rs,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append((rs, acc))

best_rs, best_acc = max(accuracies, key=lambda x: x[1])
print(f"Best random_state: {best_rs}, Accuracy: {best_acc:.4f}")
print(f"Average accuracy across runs: {np.mean([a for _, a in accuracies]):.4f}")

import matplotlib.pyplot as plt

states, accs = zip(*accuracies)
plt.plot(states, accs, marker="o")
plt.xlabel("random_state")
plt.ylabel("Accuracy")
plt.title("Accuracy variation across random splits")
plt.show()