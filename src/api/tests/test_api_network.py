import socket
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from src.api import create_app
from src.api.network import diagnose

V4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("131.234.240.215", 443))
V6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:638:502:c050::215", 443, 0, 0))


class _Socket:
    def close(self):
        pass


def test_diagnose_reports_each_address_and_treats_401_as_reachable():
    def connect(address, timeout):
        if ":" in address[0]:
            raise OSError(101, "Network is unreachable")
        return _Socket()

    def open_url(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    [report] = diagnose(
        ["https://ai-gateway.uni-paderborn.de/v1", "https://ai-gateway.uni-paderborn.de/v1"],
        resolve=lambda *args: [V6, V4],
        connect=connect,
        open_url=open_url,
    )
    by_family = {entry["family"]: entry["tcp"] for entry in report["addresses"]}
    assert by_family["IPv4"] == "connected"
    assert "Network is unreachable" in by_family["IPv6"]
    assert report["https"]["status"] == 401


def test_diagnose_reports_a_blocked_gateway():
    def connect(address, timeout):
        raise TimeoutError("timed out")

    def open_url(request, timeout):
        raise TimeoutError("timed out")

    [report] = diagnose(["https://gateway.example/v1"], resolve=lambda *args: [V4], connect=connect, open_url=open_url)
    assert report["addresses"][0]["tcp"].startswith("TimeoutError")
    assert report["https"]["error"].startswith("TimeoutError")


def test_diagnose_endpoint_never_returns_the_key(monkeypatch):
    import src.api.app as app_module

    monkeypatch.setenv("UPB_AI_GATEWAY_KEY", "secret-value")
    monkeypatch.setattr(app_module, "diagnose", lambda urls: [{"base_url": url} for url in urls])
    body = TestClient(create_app(object())).get("/api/diagnose").json()
    assert body["key_configured"]["UPB_AI_GATEWAY_KEY"] is True
    assert "secret-value" not in str(body)
