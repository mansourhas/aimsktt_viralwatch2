import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib


print("Loading data...")


data = pd.read_csv(
    "data/processed/dataset.csv"
)


X = data.drop("target", axis=1)

y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training model...")


model = RandomForestClassifier(
    n_estimators=100
)


model.fit(
    X_train,
    y_train
)


prediction = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "Accuracy:",
    accuracy
)


joblib.dump(
    model,
    "models/bdbv_model.pkl"
)


print("Model saved")