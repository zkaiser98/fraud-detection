from risk_rules import label_risk, score_transaction


# --- helpers ---

def base_tx(**overrides):
    """A clean, low-risk transaction baseline."""
    tx = {
        "device_risk_score": 5,
        "is_international": 0,
        "amount_usd": 20,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# --- label thresholds ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


# --- amount ---

def test_large_amount_adds_risk():
    assert score_transaction(base_tx(amount_usd=1200)) >= 25


def test_medium_amount_adds_some_risk():
    low = score_transaction(base_tx(amount_usd=20))
    mid = score_transaction(base_tx(amount_usd=600))
    high = score_transaction(base_tx(amount_usd=1200))
    assert low < mid < high


# --- device risk ---

def test_high_device_risk_increases_score():
    clean = score_transaction(base_tx(device_risk_score=5))
    risky = score_transaction(base_tx(device_risk_score=75))
    assert risky > clean


def test_very_high_device_risk_adds_more_than_medium():
    medium_device = score_transaction(base_tx(device_risk_score=50))
    high_device = score_transaction(base_tx(device_risk_score=75))
    assert high_device > medium_device


# --- international ---

def test_international_increases_score():
    domestic = score_transaction(base_tx(is_international=0))
    international = score_transaction(base_tx(is_international=1))
    assert international > domestic


# --- velocity ---

def test_high_velocity_increases_score():
    low_vel = score_transaction(base_tx(velocity_24h=1))
    high_vel = score_transaction(base_tx(velocity_24h=8))
    assert high_vel > low_vel


def test_velocity_tiers_ordered():
    v1 = score_transaction(base_tx(velocity_24h=1))
    v4 = score_transaction(base_tx(velocity_24h=4))
    v8 = score_transaction(base_tx(velocity_24h=8))
    assert v1 < v4 < v8


# --- failed logins ---

def test_high_failed_logins_increases_score():
    clean = score_transaction(base_tx(failed_logins_24h=0))
    suspicious = score_transaction(base_tx(failed_logins_24h=6))
    assert suspicious > clean


# --- prior chargebacks ---

def test_prior_chargebacks_increase_score():
    no_cb = score_transaction(base_tx(prior_chargebacks=0))
    one_cb = score_transaction(base_tx(prior_chargebacks=1))
    two_cb = score_transaction(base_tx(prior_chargebacks=2))
    assert no_cb < one_cb < two_cb


# --- end-to-end risk profiles ---

def test_clean_transaction_scores_low():
    score = score_transaction(base_tx())
    assert label_risk(score) == "low"


def test_known_fraud_profile_scores_high():
    # Mirrors transaction 50011: RU, high device risk, high velocity, many failed logins
    tx = base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=1,
    )
    assert label_risk(score_transaction(tx)) == "high"


def test_score_clamped_between_0_and_100():
    # Worst-case inputs should not exceed 100
    worst = base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=5000,
        velocity_24h=10,
        failed_logins_24h=10,
        prior_chargebacks=5,
    )
    assert 0 <= score_transaction(worst) <= 100

    # Best-case inputs should not go below 0
    best = base_tx()
    assert score_transaction(best) >= 0
