import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

from src.features.build_features import get_feature_lists

base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "data" / "processed" / "telco_clean.csv"

def main():
    df = pd.read_csv(data_path)

    y = df['Churn'].map({'Yes':1, 'No': 0 })
    X, cat_cols, num_cols = get_feature_lists(df)

    preprocessor = ColumnTransformer (
        transformers = [
            ('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators = 200,
        random_state = 42,
        class_weight = 'balanced'
    )

    pipeline = Pipeline(steps = [
        ('preprocess', preprocessor),
        ('model', model)
      ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, stratify = y, random_state = 31
    )

    pipeline.fit(X_train, y_train)
    proba =  pipeline.predict_proba(X_test)[:,1]
    print("ROC AUC:", round(roc_auc_score(y_test, proba), 2))

    output_dir = Path(__file__).resolve().parents[2] / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "churn_pipeline.pkl")

if __name__ == '__main__':
    main()