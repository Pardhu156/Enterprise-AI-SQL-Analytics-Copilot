import pytest


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": 123},
        {"question": "x" * 2001},
    ],
)
def test_invalid_question_returns_structured_422(app_client, payload) -> None:
    client, service = app_client
    response = client.post("/api/v1/analytics/query", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert response.json()["error"]["request_id"]
    assert not service.calls


def test_malformed_json_returns_structured_422(app_client) -> None:
    client, _ = app_client
    response = client.post(
        "/api/v1/analytics/query",
        content="{not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"

