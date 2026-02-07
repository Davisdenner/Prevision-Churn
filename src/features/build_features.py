import pandas as pd

def get_feature_lists(df:pd.DataFrame):
    target = "Churn"
    X = df.drop(columns = [target, "customerID"])
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    return X, cat_cols, num_cols