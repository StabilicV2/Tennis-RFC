#cSpell:disable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib

model_path = "../results/final_tennis_model.pkl"
model, threshold = joblib.load(model_path)

feature_names = list(model.feature_names_in_)

feature_importance = model.feature_importances_

indices = np.argsort(feature_importance)[::-1]
features = [feature_names[i] for i in indices]
sort = feature_importance[indices]

plt.figure(figsize=(12, 10))
plt.gcf().canvas.manager.set_window_title("Random Forest Feature Importance")
graph = sns.barplot(x=sort, y=features, palette="viridis")

plt.subplots_adjust(left=0.16)

for i, value in enumerate(sort):
  graph.text(value + 0.0025, i, f"{value:.3f}", ha="left", va="center", fontsize=10, color="black")

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

