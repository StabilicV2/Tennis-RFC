#cSpell:disable

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv(f"../data/preprocessed/preprocessed_all.csv")

X = df.drop(columns=["player_id", "opponent_id", "winner"])
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

model_path = "../results/small_tennis_model.pkl"
model = joblib.load(model_path)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()