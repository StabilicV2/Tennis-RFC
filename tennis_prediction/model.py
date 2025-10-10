import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
  with open("settings.txt", "r") as file:
    lines = file.readlines()

  a = lines[0].strip()
  b = lines[1].strip()

  if a == b:
    df = pd.read_csv(f"data/preprocessed/preprocessed_{a}.csv")
  
  if int(a) == 1985 and int(b) == 2024:
    df = pd.read_csv("data/preprocessed/preprocessed_all.csv")
  else:
    df = pd.read_csv(f"data/preprocessed/preprocessed_{a}_{b}.csv")

  X = df.drop(columns=["player_id", "opponent_id", "winner", "tourney_date", "date"])
  y = df["winner"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

  model = RandomForestClassifier(n_estimators=186, random_state=None, max_depth=20, min_samples_split=5, min_samples_leaf=1, max_features="log2", class_weight="balanced")
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  print(f"Accuracy: {accuracy:.4f}")

  joblib.dump((model, 0.5), "results/final_tennis_model.pkl")

  with open("settings.txt", "w") as file:
    file.write(f"{a}\n")
    file.write(f"{b}\n")
    file.write(f"{accuracy:.4f}")

if __name__ == "__main__":
  train()