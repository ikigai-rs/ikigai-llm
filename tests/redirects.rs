//! A provider's 3xx is refused, never parsed and never followed (ledger #3471).
//!
//! Host transports return a 3xx as-is (ikigai-http's contract: following is the
//! endpoint's job, under the per-hop capability ACL). This crate calls the
//! transport directly, so a redirect reaches it. It must not go on to body parsing
//! (where it used to fail as an unrelated "was not JSON" decode error), and it must
//! not be followed here either: a POST body is never replayed at a host the caller
//! did not vet. The refusal names the status and the `Location`.
//!
//! No real model is called: [`Redirect`] is an in-process [`HttpTransport`] that
//! answers every request with one redirect and records every URL it was sent.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Error, FailureClass, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig, Registry};

const ELSEWHERE: &str = "http://elsewhere.example/v1/chat/completions";

struct Redirect {
    status: u16,
    location: Option<&'static str>,
    sent: Mutex<Vec<String>>,
}

#[async_trait]
impl HttpTransport for Redirect {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        self.sent.lock().unwrap().push(request.url);
        Ok(HttpResponse {
            status: self.status,
            headers: self
                .location
                .map(|l| vec![("Location".to_string(), l.to_string())])
                .unwrap_or_default(),
            // A body that would decode, so only the status can refuse it.
            body: br#"{"choices":[{"message":{"content":"moved"}}],"data":[]}"#
                .to_vec()
                .into(),
        })
    }
}

/// Issue `target` (with `prompt` and a pinned `model` when it is an ask) against a
/// backend that answers every request `status` with `location`; return the outcome
/// and every URL the transport was sent.
fn issue(
    target: &str,
    status: u16,
    location: Option<&'static str>,
) -> (Result<Vec<u8>, Error>, Vec<String>) {
    let stub = Arc::new(Redirect {
        status,
        location,
        sent: Mutex::new(Vec::new()),
    });
    let kernel = Kernel::new(Arc::new(space(
        stub.clone(),
        Registry::single(OpenAiConfig::ollama("m")),
    )));
    let mut request = Request::new(Verb::Source, Iri::parse(target).unwrap());
    if target.ends_with(":ask") {
        request = request
            .with_arg("prompt", ArgRef::Inline(b"hi".to_vec()))
            .with_arg("model", ArgRef::Inline(b"m".to_vec()));
    }
    let outcome = futures::executor::block_on(kernel.issue(request, &Capability::root()))
        .map(|repr| repr.bytes.to_vec());
    let sent = stub.sent.lock().unwrap().clone();
    (outcome, sent)
}

fn assert_refused(target: &str, status: u16, err: &Error, sent: &[String]) {
    let text = err.to_string();
    assert!(
        matches!(err, Error::Endpoint(_)),
        "{target} {status}: a redirect is the endpoint's typed refusal, got {err:?}"
    );
    assert_eq!(err.class(), FailureClass::Unknown, "{target} {status}");
    assert!(
        text.contains(&format!("returned {status} redirect")),
        "{target} {status}: the status is named: {text}"
    );
    assert!(
        text.contains(&format!("Location: {ELSEWHERE}")),
        "{target} {status}: the Location is named: {text}"
    );
    assert!(
        text.contains("not followed"),
        "{target} {status}: says it was not followed: {text}"
    );
    assert!(
        !text.contains("not JSON") && !text.contains("choices"),
        "{target} {status}: the body was not parsed: {text}"
    );
    assert!(
        sent.iter().all(|url| !url.contains("elsewhere.example")),
        "{target} {status}: the Location was never requested: {sent:?}"
    );
}

#[test]
fn a_redirected_ask_is_refused_naming_status_and_location() {
    for status in [302, 307] {
        let (outcome, sent) = issue("urn:llm:ollama:ask", status, Some(ELSEWHERE));
        let err = outcome.expect_err("a 3xx is not an answer");
        assert_refused("ask", status, &err, &sent);
        assert_eq!(sent.len(), 1, "one POST, never replayed: {sent:?}");
        assert!(sent[0].ends_with("/chat/completions"), "{sent:?}");
    }
}

#[test]
fn a_redirected_model_listing_is_refused_naming_status_and_location() {
    for status in [302, 307] {
        let (outcome, sent) = issue("urn:llm:ollama:installed", status, Some(ELSEWHERE));
        let err = outcome.expect_err("a 3xx is not a model list");
        assert_refused("installed", status, &err, &sent);
    }
}

#[test]
fn every_redirect_status_is_refused() {
    for status in [300, 301, 302, 303, 307, 308] {
        let (outcome, sent) = issue("urn:llm:ollama:ask", status, Some(ELSEWHERE));
        assert_refused("ask", status, &outcome.unwrap_err(), &sent);
    }
}

#[test]
fn a_redirect_without_location_is_still_refused() {
    let (outcome, _) = issue("urn:llm:ollama:ask", 302, None);
    let text = outcome.unwrap_err().to_string();
    assert!(text.contains("returned 302 redirect"), "{text}");
    assert!(text.contains("no Location"), "{text}");
}

#[test]
fn a_redirected_liveness_probe_is_down() {
    // `up` reads only the status: a backend that points elsewhere is not answering.
    let (outcome, sent) = issue("urn:llm:ollama:up", 302, Some(ELSEWHERE));
    assert_eq!(outcome.unwrap(), b"false");
    assert_eq!(sent.len(), 1, "{sent:?}");
}
