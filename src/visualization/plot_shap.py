import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "data" / "processed" / "telco_clean.csv"
model_path = base_path / "models" / "churn_pipeline.pkl"
output_path = base_path / "reports" / "figures" / "shap_summary.png"


def main():
    print("Iniciando processo SHAP")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn", "customerID"])

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed_df)

    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    shap.summary_plot(shap_values_plot, X_transformed_df, show=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.show()

    print("Processo SHAP finalizado")


if __name__ == "__main__":
    main()