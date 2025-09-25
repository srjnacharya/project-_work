1.Employee Attrition Analysis & HR Dashboard

Project Title:
Employee Attrition Analysis and HR Dashboard using Machine Learning

Objective / Problem Statement:
The project aims to analyze employee attrition in an organization and build a predictive model to identify employees at risk of leaving. The dashboard provides HR teams with key insights and allows them to proactively manage attrition.

Key Goals:

* Understand patterns and trends in employee attrition.
* Build predictive models using Logistic Regression and Random Forest.
* Provide an interactive dashboard to visualize KPIs and predict individual attrition risk.

Tools / Technologies Used:

* Programming Language: Python 3.x
* Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, streamlit
* Machine Learning Models: Logistic Regression, Random Forest Classifier
* Data Visualization: Matplotlib, Seaborn, Streamlit charts
* Version Control / Deployment: GitHub or LinkedIn

Dataset:

* Dataset Name: WA_Fn-UseC_-HR-Employee-Attrition.csv
* Source: Provided in internship learning material
* Description: Contains HR information for employees including demographics, job role, salary, performance, and attrition status.

Steps Followed:

1. Data Loading and Exploration

   * Load CSV using pandas.
   * Inspect columns, data types, missing values, and basic statistics.
   * Compute KPIs: total employees, attrition rate, average age, average monthly income.

2. Exploratory Data Analysis (EDA)

   * Visualize attrition by department using stacked bar charts.
   * Examine correlations between numerical features.

3. Data Preprocessing

   * Drop unnecessary columns (EmployeeNumber, EmployeeCount, StandardHours, Over18).
   * Encode categorical variables using OneHotEncoder.
   * Scale numerical features with StandardScaler.
   * Split data into features (X) and target (y).

4. Model Building

   * Logistic Regression and Random Forest pipelines with preprocessing.
   * Address class imbalance using class_weight='balanced'.
   * Hyperparameter tuning for Random Forest using GridSearchCV.

5. Model Evaluation

   * Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
   * Select the best model based on ROC-AUC.

6. Feature Importance

   * Extract feature importance for Random Forest.
   * Extract top coefficients for Logistic Regression.

7. Dashboard

   * Build interactive Streamlit dashboard:

     * Display KPIs: total employees, attrition rate, average salary.
     * Show attrition proportion by department.
     * Display top 10 employees by predicted attrition risk.
     * Predict attrition for a single employee with user inputs.

8. Save and Load Model

   * Save the best model as best_attrition_model.pkl using joblib.
   * Load the model in the dashboard for predictions.

Sample Outputs / Screenshots:

1. KPI Metrics

   * Total Employees: 1470
   * Attrition Count: 237
   * Attrition Rate: 16.12%
   * Avg Monthly Income: $6500

2. Attrition by Department

   * Refer to attrition_by_department.png

3. Top 10 Employees by Predicted Attrition Risk

   | EmployeeNumber | Attrition_Prob | Department | JobRole    | Age | MonthlyIncome |
   | -------------- | -------------- | ---------- | ---------- | --- | ------------- |
   | 1001           | 0.87           | Sales      | Sales Exec | 28  | 4500          |
   | 1054           | 0.82           | R&D        | Lab Tech   | 32  | 5000          |

4. Sample Individual Prediction
   Predicted probability of attrition: 75%
   Predicted class: Yes

Learnings / Challenges:

* Learned how to handle categorical and numerical features using ColumnTransformer.
* Gained experience in model evaluation metrics and ROC-AUC interpretation.
* Developed an interactive Streamlit dashboard for real-time predictions.
* Challenge: Handling categorical encoding differences between scikit-learn versions (sparse vs sparse_output).
* Challenge: Balancing model performance on imbalanced datasets.

How to Run:

1. Run Analysis (Training + Evaluation)
   python employee_attrition_analysis_and_dashboard.py --run-analysis --csv "WA_Fn-UseC_-HR-Employee-Attrition.csv"

2. Run Dashboard
   streamlit run employee_attrition_analysis_and_dashboard.py



2.Project Title:
Bank Churn Prediction using Machine Learning
===========================================

Project Overview
----------------
The Bank Churn Prediction project aims to predict whether a bank customer will churn (leave the bank) based on historical customer data. By identifying potential churners, the bank can take proactive measures to improve customer retention and reduce revenue loss.

Objective
---------
- Predict customer churn using historical data.
- Identify patterns and trends in customer behavior.
- Build predictive models using Logistic Regression, Random Forest, and XGBoost.
- Provide feature insights to understand key factors influencing churn.

Key Goals
---------
- Analyze patterns and trends in customer churn.
- Build predictive models using Logistic Regression, Random Forest, and XGBoost.
- Provide feature insights using model importance.

Tools & Technologies
-------------------
- Programming Language: Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn (SMOTE)
- Machine Learning Models: Logistic Regression, Random Forest, XGBoost
- Data Visualization: Matplotlib, Seaborn
- Version Control: GitHub

Dataset
-------
- Name: BankChurners.csv
- Description: Contains customer demographic information, account details, transaction history, and churn status (Attrition_Flag).
- Number of Entries: 10,127
- Key Columns: CLIENTNUM, Attrition_Flag, Customer_Age, Gender, Income_Category, Credit_Limit, Total_Trans_Amt, etc.

Steps Followed
--------------
1. Data Loading & Exploration
   - Loaded CSV using pandas.
   - Inspected columns, data types, missing values, and basic statistics.
   - Computed churn rate:
       * Existing Customer: 83.93%
       * Attrited Customer: 16.07%

2. Exploratory Data Analysis (EDA)
   - Visualized churn distribution using count plots.
   - Analyzed age vs churn using boxplots.
   - Visualized income category vs churn.
   - Plotted correlation heatmap for numeric features.

3. Data Preprocessing
   - Separated numerical and categorical features.
   - Scaled numerical features using StandardScaler.
   - Encoded categorical features using OneHotEncoder.
   - Prepared preprocessing pipeline using ColumnTransformer.

4. Train-Test Split
   - Split dataset into 80% train and 20% test sets with stratification on Attrition_Flag.

5. Model Building
   - Built three models using pipelines with SMOTE for class imbalance:
       * Logistic Regression
       * Random Forest Classifier
       * XGBoost Classifier
   - Fitted models on training data.

6. Model Evaluation
   - Evaluated models using Accuracy, Recall, ROC-AUC, and Confusion Matrix.
   - Logistic Regression, Random Forest, and XGBoost achieved near-perfect results on this dataset.

Feature Importance
------------------
Top features identified from XGBoost:
- Total_Trans_Amt
- Months_on_book
- Customer_Age
- Total_Relationship_Count
- Credit_Limit

Sample Outputs
--------------
Logistic Regression Results:
- Accuracy: 1.0
- Recall: 1.0
- ROC-AUC: 1.0

Random Forest Results:
- Accuracy: 1.0
- Recall: 1.0
- ROC-AUC: 1.0

XGBoost Results:
- Accuracy: 0.9995
- Recall: 0.9969
- ROC-AUC: 1.0

Confusion Matrix:
Actual \ Predicted       | Existing Customer | Attrited Customer
------------------------|-----------------|-----------------
Existing Customer       | 1701            | 0
Attrited Customer       | 0               | 325

Learnings & Challenges
----------------------
- Learned to handle class imbalance with SMOTE.
- Gained experience preprocessing numerical and categorical features using pipelines.
- Interpreted feature importance to identify factors affecting churn.
- Challenge: Preventing overfitting due to near-perfect accuracy; requires careful validation on new data.

