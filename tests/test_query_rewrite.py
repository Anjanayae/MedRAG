import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import query_rewrite  # noqa: E402


def test_rewrite_query_fails_open_on_error(monkeypatch):
    """If the Groq call fails for any reason, rewrite_query must return the
    ORIGINAL query, never raise or return empty — this step should never
    be able to block retrieval."""
    def boom():
        raise RuntimeError("simulated API failure")

    import generation
    monkeypatch.setattr(generation, "get_groq_client", boom)

    result = query_rewrite.rewrite_query("how does B12 play a role in hairfall")
    assert result == "how does B12 play a role in hairfall"


def test_rewrite_query_returns_rewritten_text_on_success(monkeypatch):
    class FakeMessage:
        content = "How does vitamin B12 deficiency contribute to hair loss?"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    import generation
    monkeypatch.setattr(generation, "get_groq_client", lambda: FakeClient())

    result = query_rewrite.rewrite_query("how does B12 play a role in hairfall")
    assert result == "How does vitamin B12 deficiency contribute to hair loss?"


def test_rewrite_query_falls_back_to_original_on_empty_response(monkeypatch):
    class FakeMessage:
        content = "   "  # empty/whitespace-only response

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    import generation
    monkeypatch.setattr(generation, "get_groq_client", lambda: FakeClient())

    result = query_rewrite.rewrite_query("original query")
    assert result == "original query"