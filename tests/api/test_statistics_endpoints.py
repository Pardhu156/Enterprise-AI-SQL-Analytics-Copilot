from src.api.dependencies import APISettings
from src.api.main import create_app


def test_statistical_endpoints_run_real_calculations() -> None:
    from fastapi.testclient import TestClient

    app = create_app(APISettings("127.0.0.1", 8000, ("http://localhost:8501",)))
    with TestClient(app) as client:
        hypothesis = client.post(
            "/api/v1/statistics/hypothesis-test",
            json={
                "analysis_type": "proportion",
                "successes_a": 120,
                "trials_a": 1000,
                "successes_b": 80,
                "trials_b": 1000,
            },
        )
        interval = client.post(
            "/api/v1/statistics/confidence-interval",
            json={"estimand": "mean", "values": [8, 9, 10, 11, 12]},
        )
        experiment = client.post(
            "/api/v1/statistics/ab-test",
            json={
                "metric_type": "conversion",
                "control_successes": 80,
                "control_trials": 1000,
                "treatment_successes": 100,
                "treatment_trials": 1000,
            },
        )
        sample_size = client.post(
            "/api/v1/statistics/sample-size",
            json={
                "metric_type": "proportion",
                "baseline": 0.08,
                "minimum_detectable_effect": 0.02,
            },
        )

    assert hypothesis.status_code == 200
    assert hypothesis.json()["result"]["test_used"] == "Two-proportion z-test"
    assert interval.status_code == 200
    assert interval.json()["result"]["point_estimate"] == 10
    assert experiment.status_code == 200
    assert "causation" in experiment.json()["limitations"][0]
    assert sample_size.status_code == 200
    assert sample_size.json()["result"]["required_sample_size_per_group"] > 0


def test_statistical_request_validation_rejects_missing_data(app_client) -> None:
    client, _ = app_client
    response = client.post(
        "/api/v1/statistics/hypothesis-test",
        json={"analysis_type": "continuous"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
