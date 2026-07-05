# Credit Card Fraud Detection

A classification project comparing three models for detecting fraudulent
credit card transactions, with an emphasis on picking the right evaluation
metric and the right business tradeoff, not just training a model.

## The problem

A bank processes far more legitimate transactions than fraudulent ones.
The system needs to flag likely-fraudulent transactions for review, but
flagging too aggressively means blocking real customers' legitimate
purchases, a real cost, not a free action.

## Data

The real ULB Credit Card Fraud Detection dataset (Worldline / Machine
Learning Group, Université Libre de Bruxelles), the standard, widely-cited
dataset for this problem, sourced via a public mirror since direct Kaggle
authentication wasn't available in this environment. **284,807 real
transactions** made by European cardholders over two days in September
2013, of which **492 are fraud (0.17%)**. Features `V1`-`V28` are PCA-
transformed versions of the original transaction details. The bank
anonymized the real features before publishing the data, so the model
learns from patterns in the transformed space without access to what the
original features actually represent.

## SQL

`src/load_to_sql.py` loads the data into SQLite as two related tables
(`transactions` and an `hourly_summary` aggregate) and runs queries using
window functions and a CTE, not just basic `SELECT`s:

- A rolling 3-hour average fraud rate using `AVG() OVER (... ROWS BETWEEN
  2 PRECEDING AND CURRENT ROW)`
- Ranking each fraudulent transaction's amount within its own hour using
  `RANK() OVER (PARTITION BY hour ORDER BY Amount DESC)`
- A CTE identifying hours where the fraud rate exceeded 3x the overall
  average

That last query surfaced a real finding: **hour 26 had a fraud rate of
2.05%, roughly 7.6x the overall average of 0.27%**, a genuinely useful
signal that time-of-day/session context matters, worth flagging as a
feature-engineering idea for future work (this project doesn't currently
use hour as a model feature, but the SQL exploration suggests it should).

## Models compared

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
this repo via `.gitignore`. GitHub blocks files over 100MB, and raw data
files don't belong in version control regardless of size.

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
