# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 2. Load Dataset
# ================================
df = pd.read_csv("BankChurners.csv")  # replace with your dataset filename
df.head()

# 3. EDA
# ================================
print(df.info())
print(df['Attrition_Flag'].value_counts(normalize=True))  # churn rate

# Churn distribution
sns.countplot(x='Attrition_Flag', data=df)
plt.title("Churn Distribution")
plt.show()


# Age vs Churn
sns.boxplot(x='Attrition_Flag', y='Customer_Age', data=df)
plt.title("Age vs Churn")
plt.show()

# Geography vs Churn
sns.countplot(x='Income_Category', hue='Attrition_Flag', data=df)
plt.title("Churn by Geography")
plt.show()

# ================================
numeric_df = df.select_dtypes(include=["int64", "float64"])
# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. Preprocessing
# ================================
X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

# Identify categorical and numerical features
categorical_features = ["Income_Category", "Gender"]
numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in [" CLIENTNUM"]] 

# Preprocessor
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# 5. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 6. Baseline Model: Logistic Regression
# ================================
logreg_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Fit model
logreg_model.fit(X_train, y_train)

# Predictions
y_pred = logreg_model.predict(X_test)
y_proba = logreg_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, pos_label="Attrited Customer"))  
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# 7. Random Forest Model
# ================================
rf_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]

print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred,pos_label="Attrited Customer"))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print(dict(zip(le.classes_, le.transform(le.classes_)))) 

# ================================
# 8. XGBoost Model
# ================================
xgb_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(eval_metric='logloss', random_state=42))
])

xgb_model.fit(X_train, y_train_enc)  
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Results:")
print("Accuracy:", accuracy_score(y_test_enc, y_pred))
print("Recall:", recall_score(y_test_enc, y_pred, pos_label=0))  # 0 = "Attrited Customer"
print("ROC-AUC:", roc_auc_score(y_test_enc, y_proba))

# ================================
# 9. Confusion Matrix
# ================================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  Make sure both y_test and y_pred are numeric
cm = confusion_matrix(y_test_enc, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Existing Customer", "Attrited Customer"],
            yticklabels=["Existing Customer", "Attrited Customer"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ================================
# 10. Feature Importance (from XGBoost)
# ================================
xgb_clf = xgb_model.named_steps['classifier']
importances = xgb_clf.feature_importances_

# Get feature names
ohe_features = xgb_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
all_features = numerical_features + list(ohe_features)

feat_imp = pd.DataFrame({"feature": all_features, "importance": importances})
feat_imp.sort_values("importance", ascending=False).head(10).plot(
    x="feature", y="importance", kind="barh", figsize=(8,6)
)
plt.title("Top 10 Feature Importances (XGBoost)")
plt.show()

