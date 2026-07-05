# Credit Card Fraud Detection

A classification project comparing three models for detecting fraudulent
credit card transactions.

## The Problem:

A bank processes far more legitimate transactions than fraudulent ones.
The system needs to flag likely-fraudulent transactions for review, but
flagging too aggressively means blocking real customers' legitimate
purchases which can cause major issues.

## Data:

A dataset from Kaggle called Credit Card Fraud Detection which shows 
anonymized credit card transactions labeled as fraudulent or genuine.
"The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions."

## Models Compared:

Three classifiers were trained and evaluated, not just one: logistic
regression, random forest, and XGBoost, three of the most commonly used
classification algorithms in data analysis and machine learning.

**Accuracy is not used as the evaluation metric.** With 99.83% of
transactions being legitimate, a model that predicts "not fraud" for every
transaction would score 99.83% accuracy while catching zero fraud.
Precision and recall are used instead.

### Results (on held-out 20% test set)

| Model | Precision | Recall | F1 | ROC-AUC | False Positives | False Negatives |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.066 | 0.929 | 0.123 | 0.977 | 1,296 | 7 |
| Random Forest | 0.931 | 0.827 | 0.876 | 0.963 | 6 | 17 |
| XGBoost | 0.885 | 0.867 | 0.876 | 0.977 | 11 | 13 |

**Logistic regression (with balanced class weights) catches 92.9% of
fraud, but at a real cost: only 6.6% of its fraud flags are actually
fraud**, it would block 1,296 legitimate transactions to catch 91 real
fraud cases in this test set alone. Random forest is the opposite extreme,
with only 6 false positives total, but it misses more real fraud (82.7% recall
vs. logistic regression's 92.9%). XGBoost lands between the two, with the
best ROC-AUC of the three.

**Which model to actually deploy depends on a real business tradeoff**,
not just "which number is highest." If the cost of a false positive is low
(e.g. the customer gets a text to verify a $40 purchase) but the cost of a
false negative is high (a large fraudulent charge goes through
unchallenged), logistic regression's aggressive recall may be worth its
poor precision. If false positives carry real cost (blocked legitimate
high-value transactions, support burden, customer trust), random forest or
XGBoost's better precision is the better choice. This project doesn't
assume one answer, it presents the tradeoff explicitly.

## Project structure

```
fraud-detection/
├── data/
│   ├── creditcard.csv          # real ULB dataset
│   ├── fraud.db                 # SQLite database (transactions + hourly_summary)
│   └── model_comparison.csv     # output: precision/recall/F1/AUC per model
├── src/
│   ├── load_to_sql.py           # SQL loading + window function/CTE queries
│   └── model.py                 # trains and compares the three classifiers
└── README.md
```

## Running it

The raw dataset (150MB) and generated SQLite database are excluded from
this repo via `.gitignore`.
```bash
pip install pandas scikit-learn xgboost

# download creditcard.csv from Kaggle and place it in data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

cd fraud-detection
python src/load_to_sql.py   # loads data into SQLite, runs SQL exploration
python src/model.py          # trains and evaluates all three models
```

## What I'd build next

- Add `hour` (and hour-level fraud-rate context, surfaced by the SQL
  exploration above) as an actual model feature, not just an exploratory
  finding
- Tune the classification threshold per model rather than using the
  default 0.5, since the precision/recall tradeoff can be shifted along
  each model's ROC curve depending on the business cost assumptions above
- Test a cost-weighted evaluation (assign a dollar cost to false positives
  and false negatives) instead of comparing precision/recall in the
  abstract, to make the "which model to deploy" question quantitative
  rather than qualitative
