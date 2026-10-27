//! Incremental OpenAI-compatible SSE completions (including Ollama and mlx-lm).
//!
//! The HTTP reader, UTF-8 decoder, SSE parser and JSON parser are shared library
//! implementations. Only the provider's delta/done protocol is interpreted here.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::{io::AsyncReadExt, stream, StreamExt};
use ikigai_core::{Body, Bytes, Error};
use ikigai_http::HttpBody;
use serde_json::Value;

// Token events are small. Refuse oversized framing instead of letting the SSE
// parser accumulate an unbounded line, event, id, or comment-only response.
const MAX_EVENT_BYTES: usize = 64 * 1024;

/// Bound the wire bytes consumed without a dispatched event. This bounds a
/// malformed unterminated line/event too; a valid feed can run indefinitely.
/// Read at most 8 KiB ahead. The consumer dropping the body drops the reader.
pub(crate) fn completion(body: HttpBody) -> Body {
    let pending = Arc::new(AtomicUsize::new(0));
    let counted = pending.clone();
    let chunks = stream::try_unfold(body.into_reader(), move |mut reader| {
        let counted = counted.clone();
        async move {
            let remaining = MAX_EVENT_BYTES.saturating_sub(counted.load(Ordering::Relaxed));
            let mut chunk = vec![0; 8192.min(remaining + 1)];
            let n = reader.read(&mut chunk).await.map_err(|e| {
                ikigai_http::status::transport_error(
                    &e.to_string(),
                    "response body",
                    "llm: reading completion stream",
                )
            })?;
            if n == 0 {
                return Ok(None);
            }
            if n > remaining {
                return Err(Error::OverBudget {
                    detail: format!("llm: completion event exceeds {MAX_EVENT_BYTES} wire bytes"),
                });
            }
            counted.fetch_add(n, Ordering::Relaxed);
            chunk.truncate(n);
            Ok(Some((chunk, reader)))
        }
    });
    let events = Box::pin(chunks.eventsource());
    Body::once(stream::try_unfold(
        Some((events, pending)),
        |state| async move {
            let Some((mut events, pending)) = state else {
                return Ok(None);
            };
            let event = match events.next().await {
                Some(Ok(event)) => event,
                Some(Err(EventStreamError::Transport(error))) => return Err(error),
                // Parser diagnostics may quote a provider's echoed prompt. Never
                // copy that text into an error that travels through logs/traces.
                Some(Err(_)) => {
                    return Err(Error::Endpoint(
                        "llm: malformed completion event stream".into(),
                    ))
                }
                None => {
                    return Err(Error::Unavailable(
                        "llm: completion stream ended before [DONE]".into(),
                    ))
                }
            };
            pending.store(0, Ordering::Relaxed);
            if event.event == "error" {
                return Err(Error::Endpoint(
                    "llm: provider reported a streaming error".into(),
                ));
            }
            if event.data == "[DONE]" {
                // Let core's deadline wrapper check the completion marker too:
                // a late [DONE] must not turn a deadline refusal into clean EOF.
                return Ok(Some((Bytes::new(), None)));
            }
            let value: Value = serde_json::from_str(&event.data)
                .map_err(|_| Error::Endpoint("llm: completion event was not JSON".into()))?;
            if value.get("error").is_some() {
                return Err(Error::Endpoint(
                    "llm: provider reported a streaming error".into(),
                ));
            }
            // Usage-only events and role/finish deltas carry no completion text.
            let choices = value["choices"]
                .as_array()
                .ok_or_else(|| Error::Endpoint("llm: completion event has no choices".into()))?;
            if choices.is_empty() && value.get("usage").is_some() {
                return Ok(Some((Bytes::new(), Some((events, pending)))));
            }
            let choice = choices
                .first()
                .ok_or_else(|| Error::Endpoint("llm: completion event has no choice".into()))?;
            let delta = choice["delta"]
                .as_object()
                .ok_or_else(|| Error::Endpoint("llm: completion choice has no delta".into()))?;
            if delta
                .get("refusal")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty())
            {
                return Err(Error::Refused(
                    "llm: provider refused the completion".into(),
                ));
            }
            if delta.contains_key("tool_calls") || delta.contains_key("function_call") {
                return Err(Error::Endpoint("llm: non-text completion delta".into()));
            }
            let text = match delta.get("content").unwrap_or(&Value::Null) {
                Value::String(text) => Bytes::copy_from_slice(text.as_bytes()),
                Value::Null => Bytes::new(),
                _ => {
                    return Err(Error::Endpoint(
                        "llm: completion delta content is not text".into(),
                    ))
                }
            };
            // Metadata-only events also yield, allowing core to check the
            // deadline even when a provider has not generated text yet.
            Ok(Some((text, Some((events, pending)))))
        },
    ))
}
