"""Train churn prediction model"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Configure MLflow tracking
mlflow.set_tracking_uri("http://a4041c1ea48d14f5ca88951e0ba9b775-105051300.us-east-1.elb.amazonaws.com")
mlflow.set_experiment("Customer-Churn-Tracking")

# Load data
df = pd.read_csv('data/churn_data.csv')

# Features and target
features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'num_support_calls']
X = df[features]
y = df['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    n_estimators = 100
    random_state = 42

    # Log parameters to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("feature_count", len(features))

    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc_roc", auc)

    # Save model locally
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/churn_model.pkl")

    # Log scikit-learn model artifact to MLflow
    mlflow.sklearn.log_model(model, "model")
    print("Model logged to MLflow successfully!")
