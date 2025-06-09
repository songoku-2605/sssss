# 📊 EDA & Loan Default Prediction

This project involves **Exploratory Data Analysis (EDA)** and **Predictive Modeling** to assess the risk of loan default. The objective is to help financial institutions better understand customer risk and make informed decisions during loan approval.

---

## 📁 About the Project

Loan default is a major concern for banks and financial institutions. Using customer data such as demographics, income, loan amount, and credit history, we explore the dataset and build predictive models to determine the probability of loan default.

### Objectives:
- Analyze patterns and trends in loan applicant data.
- Identify key factors contributing to loan default.
- Build a machine learning model to predict loan default.

---

## 🧩 Project Modules

- `EDA_Loan_Prediction.ipynb`: Contains the entire project workflow including:
  - Data Loading
  - Data Cleaning
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - Model Training and Evaluation

---

## 📦 Requirements

To run this project, you’ll need the following libraries:

pip install -r requirements.txt

```bash

requirements.txt content:

pandas
numpy
matplotlib
seaborn
scikit-learn

```

## 🧪 Basic Example

```
# Load dataset
import pandas as pd

df = pd.read_csv("loan_data.csv")

# Check for missing values
print(df.isnull().sum())

# Train-Test Split
from sklearn.model_selection import train_test_split

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

## Features

-   Structured and Cleaned Dataset
    
-   Visual Analysis using Seaborn/Matplotlib
    
-   Classification Modeling
    
-   Feature Importance
    
-   Easy to scale and deploy

## 💡 Future Scope

-   Add support for ensemble models (e.g., Random Forest, XGBoost)
    
-   Hyperparameter tuning
    
-   Deploy using Flask/Streamlit for live predictions
    
-   Integrate with a database for real-time data analysis

## 📌 Dataset

  > Dataset assumed to be a CSV file containing details like Customer ID, Age, Gender, Marital Status, Income, Credit History, and Loan Status.


