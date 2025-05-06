# 🧠 Credit Card Default Prediction

This project is a **machine learning pipeline** that predicts the likelihood of a credit card customer defaulting on their payment. It explores a dataset of credit card clients and applies various preprocessing, visualization, and modeling techniques to build a reliable predictive model.

## 📁 Project Structure

- `Credit_Card_Default_Prediction.ipynb`: Main notebook containing data loading, EDA, preprocessing, modeling, and evaluation.
- `data/`: (Optional) Folder to place your dataset if not loaded inline.
- `models/`: (Optional) Folder to save trained models.
- `README.md`: Project overview and instructions.

---

## 📊 Dataset

The dataset used is the **UCI Credit Card Default Dataset**, which contains 30,000 customer records with features like:

- Demographic details: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- Payment history: `PAY_0` to `PAY_6`
- Bill and payment amounts: `BILL_AMT1` to `BILL_AMT6`, `PAY_AMT1` to `PAY_AMT6`
- Target variable: `default.payment.next.month`

---

## 🧼 Data Preprocessing

Key steps:
- Handling nulls and outliers
- Feature scaling using `StandardScaler`
- Encoding categorical variables
- Feature engineering for better model performance

---

## 📈 Exploratory Data Analysis

The notebook includes visualizations to:
- Understand class distribution
- Analyze correlations between features
- Examine trends in defaults vs. non-defaults

---

## 🤖 Modeling

Algorithms used:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

Metrics evaluated:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC score

---

## ✅ Results

- The best-performing model was Tuned Random forest Model with an accuracy of 89% and **ROC-AUC** of **0.926530**.
- Feature importance and hyperparameter tuning were also conducted for optimal performance.

 finalized to be my best model based on the evaluation metrics recall, F1 score, KS statistic.

Even though XGB tuned model is giving high recall, f1 score and Ks, by observing the metrics on train data and test data, the model is likely overfitting. There is too much difference in train and test, model is almost learning everything from train data. For this reason, i disregard this model.

Tuned Random forest is performing well on the chosen metrics than remaining models Logistic Regression and SVM.

---

## 🔧 Requirements

```bash
Python 3.7+
Jupyter Notebook
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/mukulgarg0097/credit-default-prediction.git
   ```
2. Navigate into the project directory:
   ```bash
   cd credit-default-prediction
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook Credit_Card_Default_Prediction.ipynb
   ```

---

## 📌 Future Work

- Integrate the model into a web app using Flask or Streamlit
- Automate model retraining with new data
- Explore deep learning models (e.g., neural networks)

---
## 👤 Author

**Mukul Garg**  
[LinkedIn](https://www.linkedin.com/in/mukulgarg0097/) | [GitHub]((https://github.com/MukulGarg0097/)
