# 📊 Telecom Customer Churn Prediction & Web App

An end-to-end Machine Learning project designed to predict customer churn in the telecommunications industry. This repository covers the complete workflow—from exploratory data analysis and class imbalance handling to automated hyperparameter optimization using Optuna and interactive web app deployment via Streamlit.

---

## 📌 Motivation & Kaggle Challenge

Customer retention is one of the primary drivers of profitability in subscription-based industries like telecommunications. Acquiring a new customer can cost significantly more than retaining an existing one. 

The primary motive for this project stems from the classic **[IBM / Kaggle Telco Customer Churn Challenge](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**. The challenge tasks data practitioners with analyzing customer attributes (contract length, payment methods, internet services, tenure, and demographic details) to build predictive models that accurately flag high-risk customers before they leave the service.

---

## 🛠️ Project Architecture & Pipeline

1. **Exploratory Data Analysis (EDA) & Data Cleaning**:
   - Cleaned missing values and whitespace in key columns such as `TotalCharges`.
   - Analyzed distributions and correlations across demographic, billing, and subscription variables.

2. **Feature Engineering & Preprocessing**:
   - **Categorical Encoding**: Handled non-numeric features using `OneHotEncoder`.
   - **Numeric Scaling**: Applied `StandardScaler` to continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`).
   - Combined preprocessors cleanly using `ColumnTransformer`.

3. **Class Imbalance Mitigation**:
   - Addressed target class skewness using **SMOTE** (Synthetic Minority Over-sampling Technique) on the training dataset to ensure fair representation of churners (`1`).

4. **Model Building & Tuning**:
   - Trained a baseline **Random Forest Classifier** achieving **~77.9% accuracy**.
   - Performed automated hyperparameter optimization via **Optuna**, optimizing for the minority class F1-Score (achieving an improved F1-Score of **0.635**).

5. **Model Serialization & Web Deployment**:
   - Serialized trained pipelines into `model.pkl` and `preprocessor.pkl`.
   - Built a real-time, interactive **Streamlit web application** for single-instance predictions in local currency (₹).

---

## 📂 Project Structure

```text
telecom-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Kaggle Telco Dataset
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb         # Data Cleaning, EDA, SMOTE & Optuna
│   ├── model.pkl                              # Serialized Random Forest Model
│   └── preprocessor.pkl                       # Serialized Preprocessing Pipeline
│
├── venv/                                      # Python Virtual Environment
├── app.py                                     # Streamlit Web Application Interface
├── requirements.txt                           # Project Dependencies
└── README.md                                  # Documentation
```

---

## ⚙️ Installation & Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Set Up Virtual Environment & Activate
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the interactive Streamlit web dashboard locally:

```bash
streamlit run app.py
```

Once executed, open your browser at `http://localhost:8501` to test custom customer parameters and inspect churn likelihood predictions in real-time.

---

## 📈 Model Performance Summary

| Metric | Initial Baseline | Post-Optuna Tuning |
| :--- | :--- | :--- |
| **Accuracy** | 77.9% | ~78.5% |
| **Churn Class F1-Score** | 0.58 | **0.635** |
| **Best Model Params** | Default RF | `n_estimators: 103, max_depth: 10, min_samples_split: 5, min_samples_leaf: 4` |
