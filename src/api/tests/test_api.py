from fastapi.testclient import TestClient

from src.api import create_app
from src.explore.service import ExploreInputError


class FakeResponse:
    def to_dict(self):
        return {"question": "q", "status": "insufficient_evidence"}


class FakeExperimental:
    def records(self):
        class R:
            def __init__(self, i):
                self.i = i

            def to_dict(self):
                return {"record_id": str(self.i)}

        return [R(i) for i in range(5)]


class FakeService:
    def __init__(self, error=None):
        self.error = error
        self.experimental_data = FakeExperimental()

    def explore(self, request):
        if self.error:
            raise self.error
        return FakeResponse()


def client(error=None):
    return TestClient(create_app(FakeService(error)))


def test_health():
    assert client().get("/health").json() == {"status": "ok"}


def test_explore_ok():
    response = client().post("/api/explore", json={"question": "q", "conditions": {"fuel": "methanol"}})
    assert response.status_code == 200
    assert response.json()["question"] == "q"


def test_explore_bad_shape_is_422():
    assert client().post("/api/explore", json={"question": "q", "bogus": 1}).status_code == 422


def test_explore_input_error_is_422():
    assert client(ExploreInputError("bad")).post("/api/explore", json={"question": "q"}).status_code == 422


def test_explore_pipeline_failure_is_502():
    assert client(RuntimeError("boom")).post("/api/explore", json={"question": "q"}).status_code == 502


def test_answer_that_fails_validation_is_502_not_a_client_error():
    from src.llm.service import LLMResponseError

    response = client(LLMResponseError("finding 1 references chunk_id 'x'")).post("/api/explore", json={"question": "q"})
    assert response.status_code == 502
    assert "failed validation" in response.json()["detail"]


def test_experiments_pagination():
    body = client().get("/api/experiments?limit=2&offset=1").json()
    assert body["total"] == 5 and len(body["records"]) == 2
    assert client().get("/api/experiments?limit=0").status_code == 422


def test_question_only_chat_request_gets_inferred_conditions():
    seen = {}

    class Spy(FakeService):
        def explore(self, request):
            seen["conditions"] = request.conditions
            return FakeResponse()

    c = TestClient(create_app(Spy()))
    assert c.post("/api/explore", json={"question": "burn times for methanol at 21% oxygen"}).status_code == 200
    assert seen["conditions"] == {"fuel": "methanol", "oxygen_concentration": 21.0}


def test_explicit_conditions_are_not_overridden():
    seen = {}

    class Spy(FakeService):
        def explore(self, request):
            seen["conditions"] = request.conditions
            return FakeResponse()

    c = TestClient(create_app(Spy()))
    c.post("/api/explore", json={"question": "methanol", "conditions": {"fuel": "heptane"}})
    assert seen["conditions"] == {"fuel": "heptane"}


def test_index_page_is_served():
    r = client().get("/")
    assert r.status_code == 200 and "FireLens" in r.text


def test_host_and_port_come_from_the_environment(monkeypatch):
    from src.api.__main__ import parse_args

    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert (parse_args([]).host, parse_args([]).port) == ("127.0.0.1", 8000)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "10000")
    assert (parse_args([]).host, parse_args([]).port) == ("0.0.0.0", 10000)
    assert parse_args(["--port", "9000"]).port == 9000


def test_tts_rejects_empty_text():
    response = client().post("/api/tts", json={"text": "  "})
    assert response.status_code == 422


def test_tts_returns_audio_and_alignment(monkeypatch):
    import src.api.app as api_app

    async def fake_generate_speech(text):
        assert text == "Hello human."
        return b"ID3mock-audio", {"characters": ["H"], "character_start_times_seconds": [0], "character_end_times_seconds": [0.1]}

    monkeypatch.setattr(api_app, "generate_speech", fake_generate_speech)
    response = client().post("/api/tts", json={"text": " Hello human. "})
    assert response.status_code == 200
    assert response.json()["audio_base64"]
    assert response.json()["alignment"]["characters"] == ["H"]


def test_tts_never_returns_credentials(monkeypatch):
    import src.api.app as api_app
    from src.tts.service import TTSConfigurationError

    monkeypatch.setenv("ELEVENLABS_API_KEY", "not-for-response")

    async def fail(text):
        raise TTSConfigurationError("Speech configuration missing.")

    monkeypatch.setattr(api_app, "generate_speech", fail)
    response = client().post("/api/tts", json={"text": "Hello."})
    assert response.status_code == 503
    assert "not-for-response" not in response.text
