import pandas as pd

df = pd.read_csv("/Users/umernasir/Downloads/Telco-Customer-Churn.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())
print(df["TotalCharges"].head(20))
print(df["TotalCharges"].str.strip().eq("").sum())
print(df[df["TotalCharges"].str.strip() == ""])
print(df.loc[df["TotalCharges"].str.strip() == "", ["tenure", "TotalCharges", "Churn"]])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].dtype)
print(df["TotalCharges"].isnull().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print(df["TotalCharges"].isnull().sum())
print(df.duplicated().sum())
X = df.drop("Churn", axis=1)
y = df["Churn"]
print("Features:", X.shape)
print("Target:", y.shape)
print("Numerical columns:")
print(df.select_dtypes(include="number").columns)
print("\nCategorical columns:")
print(df.select_dtypes(include="object").columns)
X=X.drop("customerID", axis=1)
print(X.shape)
y = y.map({"No":0,"Yes":1})
print(y.value_counts())




from sklearn.model_selection import train_test_split
print("Import successful")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
    )
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(include="object").columns
print("Numerical:")
print(numeric_features)
print("\nCategorical:")
print(categorical_features)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore")
print("Encoder ready")
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder, categorical_features)],
    remainder="passthrough"
)
print("Preprocessor ready")
preprocessor.fit(X_train)
print("Preprocessor fitted")
X_train_processed = preprocessor.transform(X_train)
print(X_train_processed.shape)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import time
import pandas as pd
scaled_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}
tree_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}
results = []
for name, model in scaled_models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    start = time.perf_counter()

    pipeline.fit(X_train_processed, y_train)

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = pipeline.predict(X_val_processed)
    probabilities = pipeline.predict_proba(X_val_processed)[:, 1]

    prediction_time = time.perf_counter() - start

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions),
        "Recall": recall_score(y_val, predictions),
        "F1": f1_score(y_val, predictions),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
        "Train Time": train_time,
        "Prediction Time": prediction_time
    })

for name, model in tree_models.items():

    start = time.perf_counter()

    model.fit(X_train_processed, y_train)

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = model.predict(X_val_processed)
    probabilities = model.predict_proba(X_val_processed)[:, 1]

    prediction_time = time.perf_counter() - start
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions),
        "Recall": recall_score(y_val, predictions),
        "F1": f1_score(y_val, predictions),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
        "Train Time": train_time,
        "Prediction Time": prediction_time
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1",
    ascending=False
)
print(results_df.to_string(index=False))

for name, model in {**scaled_models, **tree_models}.items():

    if name in scaled_models:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    model.fit(X_train_processed, y_train)

    train_pred = model.predict(X_train_processed)
    val_pred = model.predict(X_val_processed)
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    print(f"{name}")
    print(f"Train F1:      {train_f1:.3f}")
    print(f"Validation F1: {val_f1:.3f}")
    print(f"Difference:    {train_f1 - val_f1:.3f}")
    print("-" * 40)

from sklearn.metrics import confusion_matrix
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
log_model.fit(X_train_processed, y_train)
val_pred = log_model.predict(X_val_processed)
cm = confusion_matrix(y_val, val_pred)
print(cm)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
test_pred = log_model.predict(X_test_processed)
test_prob = log_model.predict_proba(X_test_processed)[:, 1]
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall:", recall_score(y_test, test_pred))
print("F1:", f1_score(y_test, test_pred))
print("ROC-AUC:", roc_auc_score(y_test, test_prob))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
cluster_data = df[["tenure", "MonthlyCharges", "TotalCharges"]]
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_scaled)
df["Cluster"] = clusters
print("Cluster Counts:")
print(df["Cluster"].value_counts().sort_index())
print("\nInertia:")
print(kmeans.inertia_)
print("\nCentroids:")
print(kmeans.cluster_centers_)
print("\nSample:")
print(df[["tenure", "MonthlyCharges", "TotalCharges", "Cluster"]].head(10))

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_clusters = dbscan.fit_predict(cluster_scaled)
df["DBSCAN_Cluster"] = dbscan_clusters
print("Cluster Counts:")
print(df["DBSCAN_Cluster"].value_counts().sort_index())
print("\nNumber of Clusters:")
print(len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0))
print("\nNoise Points:")
print((dbscan_clusters == -1).sum())
print("\nSample:")
print(df[["tenure", "MonthlyCharges", "TotalCharges", "DBSCAN_Cluster"]].head(10))


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca_data = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]]
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_scaled)
df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]
print("Explained Variance:")
print(pca.explained_variance_ratio_)
print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum())
print("\nComponents:")
print(pca.components_)
print("\nPCA Shape:")
print(pca_result.shape)
print("\nSample:")
print(df[["PC1", "PC2"]].head(10))
print("KMEANS")
print("Clusters:", df["Cluster"].nunique())
print(df["Cluster"].value_counts().sort_index())
print("\nDBSCAN")
print("Clusters:", df["DBSCAN_Cluster"].nunique() - (1 if -1 in df["DBSCAN_Cluster"].values else 0))
print("Noise:", (df["DBSCAN_Cluster"] == -1).sum())
print("\nPCA")
print("PC1:", pca.explained_variance_ratio_[0])
print("PC2:", pca.explained_variance_ratio_[1])
print("Total:", pca.explained_variance_ratio_.sum())




