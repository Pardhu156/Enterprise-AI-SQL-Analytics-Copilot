def test_valid_question_returns_structured_response_and_invokes_service(app_client) -> None:
    client, service = app_client
    response = client.post(
        "/api/v1/analytics/query",
        json={"question": "  What is the total revenue?  "},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Total revenue is R$100.00."
    assert payload["sql"]["validation_passed"] is True
    assert payload["result"]["rows"] == [[100.0]]
    assert payload["analysis"]["result_type"] == "scalar_kpi"
    assert payload["visualization"]["chart_type"] == "kpi"
    assert payload["execution"]["total_request_time_ms"] == 2.0
    assert service.calls == [("What is the total revenue?", payload["request_id"])]


def test_response_options_are_valid_request_fields(app_client) -> None:
    client, service = app_client
    response = client.post(
        "/api/v1/analytics/query",
        json={
            "question": "What is the total revenue?",
            "include_sql": False,
            "include_rows": False,
            "include_visualization_config": False,
        },
    )
    assert response.status_code == 200
    assert len(service.calls) == 1

