from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import pandas as pd

df = pd.read_csv("../data/preprocessed/preprocessed_all.csv")
X = df.drop(columns=["player_id", "opponent_id", "winner"])
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': randint(100, 300),   
    'max_depth': [10, 20],           
    'min_samples_split': randint(2, 11),     
    'min_samples_leaf': randint(1, 5),   
    'max_features': ['sqrt', 'log2']
}

grid = RandomizedSearchCV(RandomForestClassifier(), param_dist, cv=5, scoring='accuracy', n_jobs=8, verbose=2, random_state=42)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)
print("Validation accuracy:", grid.best_score_)