import pandas as pd
import pytest
from features import build_model_frame


def make_transactions(**overrides):
    row = {
        "transaction_id": 1,
        "account_id": 100,
        "amount_usd": 50.0,
        "failed_logins_24h": 0,
        "device_risk_score": 5,
        "is_international": 0,
        "velocity_24h": 1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def make_accounts(**overrides):
    row = {
        "account_id": 100,
        "prior_chargebacks": 0,
        "is_vip": "N",
        "kyc_level": "full",
        "account_age_days": 365,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# --- is_large_amount ---

def test_is_large_amount_set_at_threshold():
    df = build_model_frame(make_transactions(amount_usd=1000), make_accounts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_set_above_threshold():
    df = build_model_frame(make_transactions(amount_usd=2500), make_accounts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_not_set_below_threshold():
    df = build_model_frame(make_transactions(amount_usd=999.99), make_accounts())
    assert df["is_large_amount"].iloc[0] == 0


# --- login_pressure ---

def test_login_pressure_none_for_zero_failures():
    df = build_model_frame(make_transactions(failed_logins_24h=0), make_accounts())
    assert df["login_pressure"].iloc[0] == "none"


@pytest.mark.parametrize("n", [1, 2])
def test_login_pressure_low_for_one_or_two_failures(n):
    df = build_model_frame(make_transactions(failed_logins_24h=n), make_accounts())
    assert df["login_pressure"].iloc[0] == "low"


@pytest.mark.parametrize("n", [3, 5, 10])
def test_login_pressure_high_for_three_or_more_failures(n):
    df = build_model_frame(make_transactions(failed_logins_24h=n), make_accounts())
    assert df["login_pressure"].iloc[0] == "high"


# --- account merge ---

def test_account_fields_joined_correctly():
    df = build_model_frame(make_transactions(), make_accounts(prior_chargebacks=3))
    assert df["prior_chargebacks"].iloc[0] == 3


def test_unmatched_account_produces_null_prior_chargebacks():
    txns = make_transactions(account_id=999)
    accounts = make_accounts(account_id=100)
    df = build_model_frame(txns, accounts)
    assert pd.isna(df["prior_chargebacks"].iloc[0])


def test_multiple_transactions_for_same_account_all_get_account_data():
    txns = pd.DataFrame([
        {"transaction_id": 1, "account_id": 100, "amount_usd": 50.0, "failed_logins_24h": 0,
         "device_risk_score": 5, "is_international": 0, "velocity_24h": 1},
        {"transaction_id": 2, "account_id": 100, "amount_usd": 200.0, "failed_logins_24h": 1,
         "device_risk_score": 10, "is_international": 0, "velocity_24h": 2},
    ])
    df = build_model_frame(txns, make_accounts(prior_chargebacks=1))
    assert (df["prior_chargebacks"] == 1).all()
