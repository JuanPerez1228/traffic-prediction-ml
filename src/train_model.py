import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
data_path = os.path.join("data", "traffic_data.csv")
df = pd.read_csv(data_path)

# Encode categorical columns (example columns)
df_encoded = pd.get_dummies(df, columns=["Traffic_Light_State", "Weather_Condition"])

# Define features and target
X = df_encoded.drop(columns=["Traffic_Condition", "Timestamp"])
y = df["Traffic_Condition"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
base_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

# Use the best tuned model for predictions
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\nClassification Report for the Best Model:")
print(classification_report(y_test, y_pred_best))

# ===== Feature Importances Section for Tuned Model =====
importances = best_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("\nFeature Importances:")
for idx in indices:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(8,5))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Confusion Matrix and Classification Report for Tuned Model
cm = confusion_matrix(y_test, y_pred_best, labels=["Low", "Medium", "High"])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Save the tuned model to a file
joblib.dump(best_model, "traffic_rf_best_model.pkl")
print("Best model saved as traffic_rf_best_model.pkl")

# Load the saved model and make predictions
loaded_model = joblib.load("traffic_rf_best_model.pkl")
loaded_predictions = loaded_model.predict(X_test)
print("\nLoaded Model Classification Report:")
print(classification_report(y_test, loaded_predictions))
