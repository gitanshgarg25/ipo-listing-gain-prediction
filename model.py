import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_data(df):
    df["Listing_Gains_Percent"] = pd.to_numeric(df["Listing_Gains_Percent"], errors="coerce")
    df["target"] = (df["Listing_Gains_Percent"] > 0).astype(int)

    numeric_df = df.select_dtypes(include=["number"]).fillna(0)
    X = numeric_df.drop(["target", "Listing_Gains_Percent"], axis=1)
    y = numeric_df["target"]

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return model, results

if __name__ == "__main__":
    df = load_data("ipo_data.csv")
    X, y = prepare_data(df)
    model, results = train_model(X, y)

    print("Model Results")
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("ROC-AUC:", results["roc_auc"])
