from unittest.mock import patch


def test_health_ok_when_db_up(client) -> None:
    for path in ("/health", "/api/v1/health"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["database"] == "ok"


def test_health_degraded_when_db_down(client) -> None:
    with patch("app.api.v1.routes.health.check_database", return_value=False):
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["database"] == "unavailable"


def test_request_id_is_generated_and_echoed(client) -> None:
    r = client.get("/health")
    assert len(r.headers["X-Request-ID"]) == 32

    r = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})
    assert r.headers["X-Request-ID"] == "trace-abc.123"


def test_unsafe_request_id_is_replaced(client) -> None:
    r = client.get("/health", headers={"X-Request-ID": "bad id; drop table"})
    assert r.headers["X-Request-ID"] != "bad id; drop table"


def test_unknown_route_uses_error_contract(client) -> None:
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "NOT_FOUND"
    assert r.json()["detail"]["request_id"] == r.headers["X-Request-ID"]


def test_web_client_is_served(client) -> None:
    r = client.get("/app/")
    assert r.status_code == 200 and "AeroGard" in r.text
    assert client.get("/app/app.js").status_code == 200
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/app/"
