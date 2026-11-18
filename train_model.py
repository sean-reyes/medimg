import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed_data/processed_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Split first, so nothing from the test set influences the fills
# Divides the total number of 1 and above (those with heart disease) to total cases to find baseline accuracy.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  #Split into 734 training 184 testing
)

numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
binary_cols = ["fbs", "exang"]   #fasting blood sugar below or above 120mg/dl
categorical_cols = ["sex", "dataset",  "cp", "restecg", "slope"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_cols),
    ("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
    ("cat", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # Turns each cat into subcolumns "0/1" for each cat, if unknown cat is present, ignore
    ]), categorical_cols),
])

models = {
    "Always guess majority": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# Compare models using only the training data
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Final check on the test set, done once
final = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
final.fit(X_train, y_train)
pred = final.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# import joblib
# joblib.dump(final, "models/model.joblib")