import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def findThreshold(input):
    df = pd.read_csv("data/preprocessed/preprocessed_all.csv")
    if all(col in df.columns for col in ["tourney_date", "date"]):
        X = df.drop(columns=["player_id", "opponent_id", "winner", "tourney_date", "date", "match_id", "pair_key"])
    else:
        X = df.drop(columns=["player_id", "opponent_id", "winner", "match_id", "pair_key"])
    y = df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=None
    )

    model_path = "results/final_tennis_model.pkl"
    model, threshold = joblib.load(model_path)

    if input is not None:
        best_threshold = float(input)
        print(f"Using {best_threshold:.4f} as threshold")
        
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= best_threshold).astype(int)
        best_accuracy = float(accuracy_score(y_test, preds))
        print(f"New accuracy: {best_accuracy:.4f}")
    else: 

        probs = model.predict_proba(X_test)[:, 1]

        thresholds = np.linspace(0, 1, 1000)
        accuracies = []

        for t in thresholds:
            preds = (probs >= t).astype(int)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)

        best_index = np.argmax(accuracies)
        best_threshold = thresholds[best_index]
        best_accuracy = accuracies[best_index]

        print(f"Best threshold for maximum accuracy: {best_threshold:.4f}")

    joblib.dump((model, best_threshold), model_path)

    with open("settings.txt", "r") as file:
        lines = file.readlines()
        
    a = lines[0].strip()
    b = lines[1].strip()

    with open("settings.txt", "w") as file:
        file.write(f"{a}\n")
        file.write(f"{b}\n")
        file.write(f"{best_accuracy:.4f}")

if __name__ == "__main__":
  findThreshold()