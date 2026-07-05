"""
loads transaction data into SQLite as two related tables
"""

import sqlite3
import pandas as pd

DB_PATH = "data/fraud.db"


def load_data():
    df = pd.read_csv("data/creditcard.csv")
    df["hour"] = (df["Time"] // 3600).astype(int)  # Time is seconds since first transaction; data spans ~48 hours

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)

    hourly = df.groupby("hour").agg(
        n_transactions=("Class", "count"),
        n_fraud=("Class", "sum"),
        avg_amount=("Amount", "mean"),
    ).reset_index()
    hourly["fraud_rate"] = hourly["n_fraud"] / hourly["n_transactions"]
    hourly.to_sql("hourly_summary", conn, if_exists="replace", index=False)

    conn.close()
    print(f"Loaded {len(df)} transactions and {len(hourly)} hourly summary rows -> {DB_PATH}")


def run_sql_exploration():
    conn = sqlite3.connect(DB_PATH)

    print("\n-- Class balance --")
    print(pd.read_sql(
        "SELECT Class, COUNT(*) as n, "
        "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 3) as pct "
        "FROM transactions GROUP BY Class", conn))

    print("\n-- Rolling 3-hour average fraud rate, via window function on hourly_summary --")
    print(pd.read_sql(
        "SELECT hour, fraud_rate, "
        "AVG(fraud_rate) OVER (ORDER BY hour ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as rolling_3hr_avg "
        "FROM hourly_summary ORDER BY hour LIMIT 10", conn))

    print("\n-- Rank each fraudulent transaction's amount within its own hour (window function) --")
    print(pd.read_sql(
        "SELECT hour, Amount, Class, "
        "RANK() OVER (PARTITION BY hour ORDER BY Amount DESC) as amount_rank_in_hour "
        "FROM transactions WHERE Class = 1 ORDER BY hour, amount_rank_in_hour LIMIT 10", conn))

    print("\n-- CTE: hours where fraud rate was more than 3x the overall average --")
    print(pd.read_sql(
        "WITH overall AS ("
        "  SELECT AVG(fraud_rate) as avg_fraud_rate FROM hourly_summary"
        ") "
        "SELECT h.hour, h.n_transactions, h.n_fraud, ROUND(h.fraud_rate, 4) as fraud_rate, "
        "ROUND(o.avg_fraud_rate, 4) as overall_avg_fraud_rate "
        "FROM hourly_summary h, overall o "
        "WHERE h.fraud_rate > 3 * o.avg_fraud_rate "
        "ORDER BY h.fraud_rate DESC LIMIT 10", conn))

    conn.close()


if __name__ == "__main__":
    load_data()
    run_sql_exploration()

