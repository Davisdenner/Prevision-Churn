import pandas as pd
def main():
    df = pd.read_csv('raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df = df.dropna()

    df.to_csv("processed/telco_clean.csv", index=False)

if __name__ == "__main__":
    main()
