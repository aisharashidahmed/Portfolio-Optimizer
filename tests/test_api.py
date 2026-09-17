"""End-to-end API tests against a real app with a throwaway database."""

import pytest

VALID_REQUEST = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
    "algorithm": "hrp",
}


class TestHealth:
    def test_health_is_public(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestAuth:
    def test_signup_returns_a_token(self, client):
        response = client.post(
            "/auth/signup", json={"email": "new@example.com", "password": "password123"}
        )
        assert response.status_code == 201
        assert response.json()["token_type"] == "bearer"

    def test_duplicate_email_is_rejected(self, client):
        payload = {"email": "dup@example.com", "password": "password123"}
        client.post("/auth/signup", json=payload)
        assert client.post("/auth/signup", json=payload).status_code == 409

    def test_login_with_correct_password(self, client):
        client.post(
            "/auth/signup", json={"email": "u@example.com", "password": "password123"}
        )
        response = client.post(
            "/auth/login", json={"email": "u@example.com", "password": "password123"}
        )
        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_login_with_wrong_password_is_rejected(self, client):
        client.post(
            "/auth/signup", json={"email": "u@example.com", "password": "password123"}
        )
        response = client.post(
            "/auth/login", json={"email": "u@example.com", "password": "WRONG"}
        )
        assert response.status_code == 401

    def test_unknown_user_gives_the_same_error_as_a_wrong_password(self, client):
        """Login must not let an attacker enumerate registered emails."""
        client.post(
            "/auth/signup", json={"email": "real@example.com", "password": "password123"}
        )
        wrong_password = client.post(
            "/auth/login", json={"email": "real@example.com", "password": "nope12345"}
        )
        no_such_user = client.post(
            "/auth/login", json={"email": "ghost@example.com", "password": "nope12345"}
        )
        assert wrong_password.status_code == no_such_user.status_code == 401
        assert wrong_password.json() == no_such_user.json()

    def test_short_password_is_rejected(self, client):
        response = client.post(
            "/auth/signup", json={"email": "a@example.com", "password": "short"}
        )
        assert response.status_code == 422

    def test_malformed_email_is_rejected(self, client):
        response = client.post(
            "/auth/signup", json={"email": "not-an-email", "password": "password123"}
        )
        assert response.status_code == 422


class TestOptimizeAuthorization:
    def test_optimize_requires_a_token(self, client):
        assert client.post("/optimize/sync", json=VALID_REQUEST).status_code == 401

    def test_garbage_token_is_rejected(self, client):
        response = client.post(
            "/optimize/sync",
            json=VALID_REQUEST,
            headers={"Authorization": "Bearer not.a.real.token"},
        )
        assert response.status_code == 401

    def test_history_requires_a_token(self, client):
        assert client.get("/history").status_code == 401


class TestOptimizeValidation:
    @pytest.mark.parametrize(
        ("field", "value", "reason"),
        [
            ("tickers", ["AAPL"], "fewer than two tickers"),
            ("tickers", ["AAPL", "not a ticker!"], "invalid symbol"),
            ("tickers", [], "empty universe"),
            ("algorithm", "magic", "unknown algorithm"),
            ("max_weight", 0.0, "zero cap"),
            ("max_weight", 1.5, "cap above 100%"),
            ("start_date", "2024-06-01", "start after end"),
            ("end_date", "2022-01-15", "window too short"),
        ],
    )
    def test_rejects_bad_input(self, client, auth_headers, field, value, reason):
        payload = {**VALID_REQUEST, field: value}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)
        assert response.status_code == 422, f"should reject {reason}"

    def test_rejects_infeasible_weight_cap(self, client, auth_headers):
        """3 assets capped at 20% can only sum to 60%, never 100%."""
        payload = {**VALID_REQUEST, "algorithm": "mvo", "max_weight": 0.20}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)
        assert response.status_code == 422
        assert "infeasible" in response.text.lower()

    def test_tickers_are_normalised_and_deduplicated(
        self, client, auth_headers, fake_market_data
    ):
        payload = {**VALID_REQUEST, "tickers": ["aapl", "  msft  ", "AAPL"]}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["tickers"] == ["AAPL", "MSFT"]


class TestOptimizeResults:
    @pytest.mark.parametrize("algorithm", ["hrp", "mvo", "equal_weight"])
    def test_weights_sum_to_one(self, client, auth_headers, fake_market_data, algorithm):
        payload = {**VALID_REQUEST, "algorithm": algorithm, "max_weight": 0.5}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)

        assert response.status_code == 200, response.text
        weights = response.json()["weights"]
        assert sum(weights) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in weights)

    def test_returns_one_weight_per_ticker(self, client, auth_headers, fake_market_data):
        response = client.post("/optimize/sync", json=VALID_REQUEST, headers=auth_headers)
        body = response.json()
        assert len(body["weights"]) == len(body["tickers"])

    def test_returns_portfolio_metrics(self, client, auth_headers, fake_market_data):
        response = client.post("/optimize/sync", json=VALID_REQUEST, headers=auth_headers)
        metrics = response.json()["metrics"]
        assert set(metrics) == {
            "expected_annual_return",
            "annual_volatility",
            "sharpe_ratio",
        }
        assert metrics["annual_volatility"] > 0

    def test_mvo_respects_the_weight_cap(self, client, auth_headers, fake_market_data):
        payload = {**VALID_REQUEST, "algorithm": "mvo", "max_weight": 0.4}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)
        assert max(response.json()["weights"]) <= 0.4 + 1e-6


class TestHistory:
    def test_history_starts_empty(self, client, auth_headers):
        response = client.get("/history", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []


class TestCovarianceEstimatorOption:
    def test_shrinkage_is_the_default(self, client, auth_headers, fake_market_data):
        response = client.post("/optimize/sync", json=VALID_REQUEST, headers=auth_headers)
        diagnostics = response.json()["covariance_diagnostics"]
        assert diagnostics["estimator"] == "ledoit_wolf"

    def test_sample_estimator_can_be_requested(
        self, client, auth_headers, fake_market_data
    ):
        payload = {**VALID_REQUEST, "covariance_estimator": "sample"}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)

        assert response.status_code == 200
        diagnostics = response.json()["covariance_diagnostics"]
        assert diagnostics["estimator"] == "sample"
        assert diagnostics["shrinkage_intensity"] == 0.0

    def test_diagnostics_report_data_adequacy(
        self, client, auth_headers, fake_market_data
    ):
        response = client.post("/optimize/sync", json=VALID_REQUEST, headers=auth_headers)
        diagnostics = response.json()["covariance_diagnostics"]

        assert diagnostics["n_assets"] == 3
        assert diagnostics["observations_per_asset"] > 0
        assert 0.0 <= diagnostics["shrinkage_intensity"] <= 1.0

    def test_unknown_estimator_is_rejected(self, client, auth_headers):
        payload = {**VALID_REQUEST, "covariance_estimator": "magic"}
        response = client.post("/optimize/sync", json=payload, headers=auth_headers)
        assert response.status_code == 422

    def test_the_two_estimators_disagree(self, client, auth_headers, fake_market_data):
        """If shrinkage changed nothing, wiring it up would be pointless."""
        shrunk = client.post(
            "/optimize/sync",
            json={**VALID_REQUEST, "covariance_estimator": "ledoit_wolf"},
            headers=auth_headers,
        ).json()
        sample = client.post(
            "/optimize/sync",
            json={**VALID_REQUEST, "covariance_estimator": "sample"},
            headers=auth_headers,
        ).json()

        assert shrunk["covariance_diagnostics"]["shrinkage_intensity"] > 0
        assert shrunk["weights"] != sample["weights"]


BACKTEST_REQUEST = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "algorithm": "hrp",
    "max_weight": 0.5,
    "training_days": 252,
}


@pytest.fixture
def fake_long_history(monkeypatch):
    """Enough synthetic history for a walk-forward backtest."""
    import numpy as np
    import pandas as pd

    def _prepare(tickers, start_date, end_date):
        rng = np.random.default_rng(7)
        n_obs = 900
        index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
        market = rng.normal(0.0004, 0.010, n_obs)
        betas = rng.uniform(0.7, 1.3, len(tickers))
        data = np.outer(market, betas) + rng.normal(0, 0.008, (n_obs, len(tickers)))
        daily = pd.DataFrame(data, index=index, columns=list(tickers))
        prices = 100 * (1 + daily).cumprod()
        return {
            "prices": prices,
            "daily_returns": daily,
            "expected_annual_returns": daily.mean() * 252,
            "annual_covariance": daily.cov() * 252,
            "missing_tickers": [],
        }

    monkeypatch.setattr("src.services.analysis.prepare_market_data", _prepare)
    return _prepare


class TestBacktestEndpoint:
    def test_requires_authentication(self, client):
        assert client.post("/backtest", json=BACKTEST_REQUEST).status_code == 401

    def test_returns_one_entry_per_strategy(
        self, client, auth_headers, fake_long_history
    ):
        response = client.post("/backtest", json=BACKTEST_REQUEST, headers=auth_headers)

        assert response.status_code == 200, response.text
        strategies = {s["strategy"] for s in response.json()["strategies"]}
        assert strategies == {"equal_weight", "hrp", "mvo"}

    def test_curves_are_downsampled_for_transport(
        self, client, auth_headers, fake_long_history
    ):
        response = client.post("/backtest", json=BACKTEST_REQUEST, headers=auth_headers)
        for strategy in response.json()["strategies"]:
            assert len(strategy["curve"]) <= 400

    def test_curve_keeps_its_final_point(self, client, auth_headers, fake_long_history):
        """Downsampling must not drop the number the reader will quote."""
        response = client.post("/backtest", json=BACKTEST_REQUEST, headers=auth_headers)
        equal = next(
            s for s in response.json()["strategies"] if s["strategy"] == "equal_weight"
        )
        final_value = equal["curve"][-1]["value"]
        implied = equal["performance"]["total_return"] + 1.0
        assert final_value == pytest.approx(implied, rel=1e-6)

    def test_can_request_a_subset_of_strategies(
        self, client, auth_headers, fake_long_history
    ):
        payload = {**BACKTEST_REQUEST, "strategies": ["equal_weight"]}
        response = client.post("/backtest", json=payload, headers=auth_headers)
        assert [s["strategy"] for s in response.json()["strategies"]] == ["equal_weight"]

    def test_rejects_a_window_too_short_to_leave_out_of_sample(
        self, client, auth_headers
    ):
        payload = {**BACKTEST_REQUEST, "end_date": "2020-06-01"}
        response = client.post("/backtest", json=payload, headers=auth_headers)
        assert response.status_code == 422

    def test_one_failing_strategy_does_not_sink_the_others(
        self, client, auth_headers, fake_long_history, monkeypatch
    ):
        """MVO can legitimately refuse; the reader still gets the rest."""
        from src.algorithms.mvo import OptimizationFailedError

        def boom(self, returns, cov):
            raise OptimizationFailedError("synthetic failure")

        monkeypatch.setattr("src.algorithms.mvo.MVOOptimizer.optimize", boom)

        response = client.post("/backtest", json=BACKTEST_REQUEST, headers=auth_headers)
        assert response.status_code == 200

        by_name = {s["strategy"]: s for s in response.json()["strategies"]}
        assert by_name["mvo"]["error"] is not None
        assert by_name["equal_weight"]["error"] is None
        assert by_name["equal_weight"]["curve"]


class TestNseEndpoint:
    def test_is_public(self, client):
        """Market data, not user data -- the landing page reads it signed out."""
        assert client.get("/nse").status_code == 200

    def test_reports_coverage_fields(self, client):
        body = client.get("/nse").json()
        assert set(body) >= {
            "tickers_known",
            "price_rows",
            "latest_trade_date",
            "usable_for_covariance",
            "listings",
        }
