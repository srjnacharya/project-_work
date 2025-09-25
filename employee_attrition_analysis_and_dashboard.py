# employee_attrition_analysis_and_dashboard.py
# Dataset path: WA_Fn-UseC_-HR-Employee-Attrition.csv
# NOTE: When running the script from the notebook or terminal, ensure the CSV path points to the above location or copy the CSV to the script folder.

"""
Employee Attrition Analysis + HR Dashboard (Streamlit)

What this file contains (step-by-step):
1) Data loading + basic checks
2) Exploratory Data Analysis (KPI calculations + sample plots)
3) Preprocessing (cleaning, encoding, scaling)
4) Build two classification models (Logistic Regression + Random Forest)
   - Use class_weight to address imbalance
   - Hyperparameter tuning for Random Forest
5) Evaluate models (classification report, ROC AUC, confusion matrix)
6) Save the best model (joblib)
7) Streamlit dashboard code that loads the CSV + saved model and shows KPIs,
   department breakdown, attrition-risk leaderboard, and an individual prediction widget.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn  # NEW: needed for version check

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, accuracy_score, precision_score, recall_score, f1_score)
import joblib


# ========================
# 1. Utility functions
# ========================
def load_data(path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """Load CSV and basic sanity checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at {path}. Please put the HR CSV in this folder.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df

def summarize_dataframe(df):
    print('\n=== Dataframe Head ===')
    print(df.head())
    print('\n=== Info ===')
    print(df.info())
    print('\n=== Missing values per column ===')
    print(df.isna().sum())


# ========================
# 2. EDA / KPIs
# ========================
def compute_kpis(df):
    kpis = {}
    total = len(df)
    kpis['total_employees'] = total
    if 'Attrition' in df.columns:
        kpis['attrition_count'] = int((df['Attrition'] == 'Yes').sum())
        kpis['attrition_rate'] = kpis['attrition_count'] / total
    else:
        kpis['attrition_count'] = None
        kpis['attrition_rate'] = None
    # Basic demographics
    if 'Age' in df.columns:
        kpis['avg_age'] = df['Age'].mean()
    if 'MonthlyIncome' in df.columns:
        kpis['avg_monthly_income'] = df['MonthlyIncome'].mean()
    return kpis

def plot_attrition_by_department(df, save=False):
    if 'Department' not in df.columns or 'Attrition' not in df.columns:
        print('Department or Attrition column missing - cannot plot attrition by department')
        return
    agg = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
    agg_prop = agg.div(agg.sum(axis=1), axis=0)
    ax = agg_prop.plot.bar(stacked=True, figsize=(8,5))
    plt.title('Proportion of Attrition by Department')
    plt.ylabel('Proportion')
    if save:
        plt.savefig('attrition_by_department.png', bbox_inches='tight')
    plt.show()


# ========================
# 3. Preprocessing & feature engineering
# ========================
def preprocess_for_model(df, drop_columns=None, target='Attrition'):
    """Return X (features) and y (binary) ready for model training."""
    df_proc = df.copy()

    default_drops = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    if drop_columns is None:
        drop_columns = default_drops
    else:
        drop_columns = list(set(default_drops + drop_columns))

    for c in drop_columns:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])

    if target not in df_proc.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    # Convert target to binary
    y = (df_proc[target] == 'Yes').astype(int)
    X = df_proc.drop(columns=[target])

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    return X, y, numerical_cols, categorical_cols


# ========================
# 4. Build & evaluate models
# ========================
def build_preprocessor(numerical_cols, categorical_cols):
    """
    Build ColumnTransformer with StandardScaler and OneHotEncoder.
    Handles sklearn version differences (sparse vs sparse_output).
    """
    num_transformer = StandardScaler()

    skl_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
    if skl_ver >= (1, 2):
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ])
    return preprocessor

def train_and_evaluate(X, y, preprocessor, random_state=42, use_gridsearch=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)

    pipe_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state))
    ])

    pipe_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=random_state))
    ])

    print('\nTraining Logistic Regression...')
    pipe_lr.fit(X_train, y_train)
    print('Done')

    print('\nTraining Random Forest...')
    pipe_rf.fit(X_train, y_train)
    print('Done')

    best_rf = pipe_rf
    if use_gridsearch:
        print('\nRunning GridSearchCV on Random Forest (small grid)')
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 6, 10]
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipe_rf, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        print('GridSearch best params:', grid.best_params_)
        best_rf = grid.best_estimator_

    results = {}
    for name, model in [('LogisticRegression', pipe_lr), ('RandomForest', best_rf)]:
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model,
            'classification_report': report,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))
        print('ROC AUC:', auc)
        print('Confusion matrix:\n', cm)

    best_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    print(f"\nSelected best model: {best_name} (ROC AUC = {results[best_name]['roc_auc']:.4f})")

    return results, best_name


# ========================
# 5. Feature importance helper
# ========================
def get_feature_names_from_preprocessor(preprocessor, numerical_cols, categorical_cols):
    feature_names = []
    feature_names.extend(numerical_cols)
    ohe = preprocessor.named_transformers_['cat']
    if hasattr(ohe, 'get_feature_names_out'):
        cat_names = ohe.get_feature_names_out(categorical_cols)
    else:
        cat_names = ohe.get_feature_names(categorical_cols)
    feature_names.extend(cat_names)
    return feature_names

def show_feature_importance(model_pipeline, numerical_cols, categorical_cols, top_n=20):
    preprocessor = model_pipeline.named_steps['preprocessor']
    clf = model_pipeline.named_steps['clf']

    try:
        feature_names = get_feature_names_from_preprocessor(preprocessor, numerical_cols, categorical_cols)
    except Exception as e:
        print('Could not get feature names: ', e)
        feature_names = None

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        fi = pd.Series(importances, index=feature_names if feature_names is not None else range(len(importances)))
        fi = fi.sort_values(ascending=False)
        print('\nTop feature importances:')
        print(fi.head(top_n))
        return fi

    elif hasattr(clf, 'coef_'):
        coefs = clf.coef_.ravel()
        coef_s = pd.Series(coefs, index=feature_names if feature_names is not None else range(len(coefs)))
        coef_s = coef_s.sort_values(key=abs, ascending=False)
        print('\nTop coefficients (absolute):')
        print(coef_s.head(top_n))
        return coef_s

    else:
        print('Model has no feature_importances_ or coef_ attribute')
        return None


# ========================
# 6. Save / Load model
# ========================
def save_model(model, filename='best_attrition_model.pkl'):
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def load_model(filename='best_attrition_model.pkl'):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None


# ========================
# 7. Analysis runner
# ========================
def run_analysis(csv_path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    df = load_data(csv_path)
    summarize_dataframe(df)

    kpis = compute_kpis(df)
    print('\nKPIs:')
    for k, v in kpis.items():
        print(f" - {k}: {v}")

    X, y, numerical_cols, categorical_cols = preprocess_for_model(df)
    print('\nNumeric cols:', numerical_cols)
    print('Categorical cols (count):', len(categorical_cols))

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    results, best_name = train_and_evaluate(X, y, preprocessor)
    best_model = results[best_name]['model']

    fi = show_feature_importance(best_model, numerical_cols, categorical_cols)
    save_model(best_model)

    print('\nAnalysis complete. You can now run the Streamlit dashboard with:')
    print('   streamlit run employee_attrition_analysis_and_dashboard.py')


# ========================
# 8. Streamlit Dashboard
# ========================
def run_dashboard(csv_path='WA_Fn-UseC_-HR-Employee-Attrition.csv', model_path='best_attrition_model.pkl'):
    try:
        import streamlit as st
    except Exception:
        print('Streamlit is required to run the dashboard. Please install streamlit and run: streamlit run employee_attrition_analysis_and_dashboard.py')
        return

    st.set_page_config(page_title='HR Attrition Dashboard', layout='wide')
    st.title('Employee Attrition — HR Dashboard')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error(f'CSV not found at {csv_path}. Place the HR dataset in the same folder.')
        return

    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    attrition_count = int((df['Attrition'] == 'Yes').sum()) if 'Attrition' in df.columns else None
    attrition_rate = (attrition_count / total) if attrition_count is not None else None

    with col1:
        st.metric('Total employees', total)
    with col2:
        st.metric('Attritions (count)', attrition_count)
    with col3:
        st.metric('Attrition rate', f"{attrition_rate:.2%}" if attrition_rate is not None else 'N/A')
    with col4:
        if 'MonthlyIncome' in df.columns:
            st.metric('Avg Monthly Income', f"{df['MonthlyIncome'].mean():.0f}")

    st.markdown('---')

    if 'Department' in df.columns and 'Attrition' in df.columns:
        dept = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        dept_prop = dept.div(dept.sum(axis=1), axis=0)
        st.subheader('Attrition proportion by Department')
        st.bar_chart(dept_prop)

    model = load_model(model_path)
    if model is None:
        st.warning('Trained model not found (best_attrition_model.pkl). KPIs will still be shown.')
    else:
        st.success('Loaded trained model: best_attrition_model.pkl')

        features_df = df.copy()
        if 'Attrition' in features_df.columns:
            features_df = features_df.drop(columns=['Attrition'])

        try:
            probs = model.predict_proba(features_df)[:, 1]
            df['Attrition_Prob'] = probs

            st.subheader('Top 10 employees by predicted attrition risk')
            if 'EmployeeNumber' in df.columns:
                top10 = df.sort_values('Attrition_Prob', ascending=False).head(10)[['EmployeeNumber','Attrition_Prob','Department','JobRole','Age','MonthlyIncome']].fillna('')
            else:
                top10 = df.sort_values('Attrition_Prob', ascending=False).head(10)
            st.dataframe(top10)

            st.subheader('Attrition probability distribution')
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(probs, bins=30)
            ax.set_xlabel('Predicted attrition probability')
            ax.set_ylabel('Count')
            st.pyplot(fig)

        except Exception as e:
            st.error(f'Could not run predictions on dataset: {e}')

        st.markdown('---')

        st.subheader('Predict for one employee (fill features)')
        with st.form('employee_form'):
            age = st.number_input('Age', min_value=18, max_value=70, value=30)
            monthly_income = st.number_input('MonthlyIncome', min_value=0, value=5000)
            job_role = st.selectbox('JobRole', options=sorted(df['JobRole'].unique())) if 'JobRole' in df.columns else st.text_input('JobRole')
            department = st.selectbox('Department', options=sorted(df['Department'].unique())) if 'Department' in df.columns else st.text_input('Department')
            submit = st.form_submit_button('Predict')

            if submit:
                sample = df.drop(columns=['Attrition']) if 'Attrition' in df.columns else df.copy()
                sample = sample.head(1).copy()
                if 'Age' in sample.columns:
                    sample.loc[:, 'Age'] = age
                if 'MonthlyIncome' in sample.columns:
                    sample.loc[:, 'MonthlyIncome'] = monthly_income
                if 'JobRole' in sample.columns:
                    sample.loc[:, 'JobRole'] = job_role
                if 'Department' in sample.columns:
                    sample.loc[:, 'Department'] = department

                try:
                    prob = model.predict_proba(sample)[:, 1][0]
                    st.write(f'Predicted probability of attrition: {prob:.2%}')
                    st.write('Predicted class:', 'Yes' if prob > 0.5 else 'No')
                except Exception as e:
                    st.error(f'Prediction failed: {e}')

    st.markdown('---')
    st.caption('Dashboard generated from employee_attrition_analysis_and_dashboard.py')


# ========================
# 9. CLI entrypoint
# ========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Employee Attrition Analysis & Dashboard runner')
    parser.add_argument('--run-analysis', action='store_true', help='Run the analysis (training + evaluation)')
    parser.add_argument('--csv', type=str, default='WA_Fn-UseC_-HR-Employee-Attrition.csv', help='Path to CSV file')
    parser.add_argument('--model', type=str, default='best_attrition_model.pkl', help='Model filename to save/load')
    args = parser.parse_args()

    if args.run_analysis:
        run_analysis(csv_path=args.csv)
    else:
        # If running with Streamlit, this will be called.
        run_dashboard(csv_path=args.csv, model_path=args.model)
