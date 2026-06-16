"""Unit tests for LangfuseTracingProcessor metadata handling."""

from collections.abc import Mapping
from datetime import datetime
from datetime import timezone
from typing import Any
from unittest.mock import MagicMock

from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.langfuse_tracing_processor import LangfuseTracingProcessor


def _make_trace(metadata: Mapping[str, Any]) -> MagicMock:
    trace = MagicMock()
    trace.trace_id = "trace-123"
    trace.name = "run_llm_loop"
    trace.export.return_value = {"metadata": metadata}
    return trace


def _make_client_with_observation() -> tuple[MagicMock, MagicMock]:
    observation = MagicMock()
    observation.trace_id = "lf-trace-1"
    observation.id = "lf-span-1"
    client = MagicMock()
    client.start_observation.return_value = observation
    return client, observation


def _make_span(span_data: GenerationSpanData) -> MagicMock:
    span = MagicMock()
    span.trace_id = "trace-123"
    span.span_id = "span-456"
    span.parent_id = None
    span.span_data = span_data
    span.started_at = "2026-06-08T12:00:00+00:00"
    span.error = None
    return span


def test_on_trace_start_promotes_user_id_and_session_id() -> None:
    """user_id and chat_session_id in metadata must be passed as first-class
    fields on update_trace so Langfuse populates the Users and Sessions views.
    """
    client, observation = _make_client_with_observation()
    processor = LangfuseTracingProcessor(client=client)

    metadata = {
        "tenant_id": "tenant-abc",
        "chat_session_id": "session-xyz",
        "user_id": "user-42",
    }
    processor.on_trace_start(_make_trace(metadata))

    observation.update_trace.assert_called_once()
    kwargs = observation.update_trace.call_args.kwargs
    assert kwargs["user_id"] == "user-42"
    assert kwargs["session_id"] == "session-xyz"
    assert kwargs["name"] == "run_llm_loop"
    assert kwargs["metadata"] == metadata


def test_on_trace_start_user_id_missing_passes_none() -> None:
    """Anonymous / unattributed traces still update successfully with user_id=None."""
    client, observation = _make_client_with_observation()
    processor = LangfuseTracingProcessor(client=client)

    metadata = {"tenant_id": "tenant-abc", "chat_session_id": "session-xyz"}
    processor.on_trace_start(_make_trace(metadata))

    kwargs = observation.update_trace.call_args.kwargs
    assert kwargs["user_id"] is None
    assert kwargs["session_id"] == "session-xyz"


def test_on_trace_start_coerces_non_string_user_id() -> None:
    """User ids that arrive as ints (e.g. from User.id) are coerced to strings."""
    client, observation = _make_client_with_observation()
    processor = LangfuseTracingProcessor(client=client)

    metadata = {"chat_session_id": "session-xyz", "user_id": 7}
    processor.on_trace_start(_make_trace(metadata))

    kwargs = observation.update_trace.call_args.kwargs
    assert kwargs["user_id"] == "7"


def test_generation_span_uses_flow_name_and_langfuse_generation_payload() -> None:
    """Generation observations should keep Onyx flow tags visible in Langfuse."""
    root_observation = MagicMock()
    root_observation.trace_id = "lf-trace-1"
    root_observation.id = "lf-root-span-1"
    generation_observation = MagicMock()
    generation_observation.id = "lf-generation-span-1"
    client = MagicMock()
    client.start_observation.side_effect = [root_observation, generation_observation]
    processor = LangfuseTracingProcessor(client=client)

    metadata = {
        "tenant_id": "tenant-abc",
        "chat_session_id": "session-xyz",
        "user_id": "user-42",
    }
    processor.on_trace_start(_make_trace(metadata))

    span_data = GenerationSpanData(
        input=[{"role": "user", "content": "hello"}],
        output=[{"role": "assistant", "content": "hi"}],
        reasoning="short reasoning",
        model="gpt-5-mini",
        model_config={"flow": "chat_response", "temperature": 0.1},
        usage={"input_tokens": 10, "output_tokens": 5},
        time_to_first_action_seconds=1.25,
    )
    span = _make_span(span_data)

    processor.on_span_start(span)

    start_kwargs = client.start_observation.call_args_list[1].kwargs
    assert start_kwargs["trace_context"] == {
        "trace_id": "lf-trace-1",
        "parent_span_id": "lf-root-span-1",
    }
    assert start_kwargs["name"] == "chat_response"
    assert start_kwargs["as_type"] == "generation"
    assert start_kwargs["metadata"] == metadata
    assert start_kwargs["model"] == "gpt-5-mini"
    assert start_kwargs["model_parameters"] == {"temperature": 0.1}

    processor.on_span_end(span)

    update_kwargs = generation_observation.update.call_args.kwargs
    assert update_kwargs["input"] == [{"role": "user", "content": "hello"}]
    assert update_kwargs["output"] == [{"role": "assistant", "content": "hi"}]
    assert update_kwargs["usage_details"] == {
        "input": 10,
        "output": 5,
        "total": 15,
    }
    assert update_kwargs["metadata"] == {
        **metadata,
        "reasoning": "short reasoning",
    }
    assert update_kwargs["completion_start_time"] == datetime(
        2026, 6, 8, 12, 0, 1, 250000, tzinfo=timezone.utc
    )
    generation_observation.end.assert_called_once()
