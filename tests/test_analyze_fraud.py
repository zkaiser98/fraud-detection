import pandas as pd
from analyze_fraud import summarize_results


def make_scored(rows):
    """rows: list of (transaction_id, account_id, amount_usd, risk_score, risk_label)"""
    return pd.DataFrame(rows, columns=["transaction_id", "account_id", "amount_usd", "risk_score", "risk_label"])


def make_chargebacks(*txn_ids):
    return pd.DataFrame({"transaction_id": list(txn_ids)})


# --- output structure ---

def test_summary_has_required_columns():
    scored = make_scored([(1, 100, 50.0, 10, "low")])
    result = summarize_results(scored, make_chargebacks())
    for col in ["risk_label", "transactions", "total_amount_usd", "avg_amount_usd", "chargebacks", "chargeback_rate"]:
        assert col in result.columns, f"Missing column: {col}"


def test_one_row_per_risk_label():
    scored = make_scored([
        (1, 100, 50.0, 10, "low"),
        (2, 101, 200.0, 70, "high"),
    ])
    result = summarize_results(scored, make_chargebacks())
    assert set(result["risk_label"]) == {"low", "high"}
    assert len(result) == 2


# --- transaction counts ---

def test_transaction_counts_per_label():
    scored = make_scored([
        (1, 100, 50.0, 10, "low"),
        (2, 101, 60.0, 15, "low"),
        (3, 102, 200.0, 70, "high"),
    ])
    result = summarize_results(scored, make_chargebacks())
    counts = dict(zip(result["risk_label"], result["transactions"]))
    assert counts["low"] == 2
    assert counts["high"] == 1


# --- amount aggregations ---

def test_total_and_avg_amount_correct():
    scored = make_scored([
        (1, 100, 100.0, 70, "high"),
        (2, 101, 300.0, 80, "high"),
    ])
    result = summarize_results(scored, make_chargebacks())
    row = result[result["risk_label"] == "high"].iloc[0]
    assert row["total_amount_usd"] == 400.0
    assert row["avg_amount_usd"] == 200.0


# --- chargeback rate ---

def test_chargeback_rate_is_fraction_of_transactions():
    scored = make_scored([
        (1, 100, 100.0, 70, "high"),
        (2, 101, 200.0, 75, "high"),
    ])
    result = summarize_results(scored, make_chargebacks(1))
    row = result[result["risk_label"] == "high"].iloc[0]
    assert row["chargebacks"] == 1
    assert row["chargeback_rate"] == 0.5


def test_no_chargebacks_gives_zero_rate():
    scored = make_scored([(1, 100, 50.0, 10, "low")])
    result = summarize_results(scored, make_chargebacks())
    row = result[result["risk_label"] == "low"].iloc[0]
    assert row["chargebacks"] == 0
    assert row["chargeback_rate"] == 0.0


def test_all_transactions_are_chargebacks_gives_rate_of_one():
    scored = make_scored([
        (1, 100, 100.0, 70, "high"),
        (2, 101, 200.0, 80, "high"),
    ])
    result = summarize_results(scored, make_chargebacks(1, 2))
    row = result[result["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == 1.0


def test_chargebacks_counted_per_label_independently():
    scored = make_scored([
        (1, 100, 50.0, 10, "low"),
        (2, 101, 200.0, 70, "high"),
        (3, 102, 300.0, 80, "high"),
    ])
    # Only the high-risk txn is a chargeback; low should still show 0
    result = summarize_results(scored, make_chargebacks(2))
    low_row = result[result["risk_label"] == "low"].iloc[0]
    high_row = result[result["risk_label"] == "high"].iloc[0]
    assert low_row["chargebacks"] == 0
    assert high_row["chargebacks"] == 1


def test_chargeback_not_in_scored_transactions_does_not_inflate_count():
    # A chargeback for a txn_id that isn't in the scored set should not affect counts
    scored = make_scored([(1, 100, 100.0, 70, "high")])
    result = summarize_results(scored, make_chargebacks(999))
    row = result[result["risk_label"] == "high"].iloc[0]
    assert row["chargebacks"] == 0
