//! What an error RESPONSE contributes to the error a caller sees (ledger #178).
//!
//! An origin's error body is a served representation like any other, and an
//! error travels further than an answer does: through traces, logs, MCP replies
//! and the far side of a mount. So a body this module cannot read as a
//! structured error is DESCRIBED (its media type and its length), never
//! forwarded: an nginx error page or a proxy's plain-text refusal is the
//! origin's text, and none of it reaches the caller.
//!
//! What still passes through, by design: the status code, and the
//! human-readable reason fields of a JSON error body (`error.message`, a bare
//! `error` string, `message`, `detail`, FastAPI's `detail[].msg`), with the
//! caller's own prompt cut out wherever a server echoed it.
//!
//! No real model is called: [`Stub`] is an in-process [`HttpTransport`] that
//! answers every request with one canned response.

use std::sync::Arc;

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig, Registry};

struct Stub(HttpResponse);

#[async_trait]
impl HttpTransport for Stub {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, String> {
        Ok(self.0.clone())
    }
}

/// The error text `urn:llm:ollama:ask` (model pinned, so no fallback probe)
/// reports when the backend answers `status` with `body` under `content_type`.
fn error_for(status: u16, content_type: Option<&str>, body: &[u8]) -> String {
    let headers = content_type
        .map(|ct| vec![("Content-Type".to_string(), ct.to_string())])
        .unwrap_or_default();
    let stub: Arc<dyn HttpTransport> = Arc::new(Stub(HttpResponse {
        status,
        headers,
        body: body.to_vec(),
    }));
    let kernel = Kernel::new(Arc::new(space(
        stub,
        Registry::single(OpenAiConfig::ollama("m")),
    )));
    let request = Request::new(Verb::Source, Iri::parse("urn:llm:ollama:ask").unwrap())
        .with_arg("prompt", ArgRef::Inline(b"hi".to_vec()))
        .with_arg("model", ArgRef::Inline(b"m".to_vec()));
    futures::executor::block_on(kernel.issue(request, &Capability::root()))
        .expect_err("a 4xx/5xx must be an error")
        .to_string()
}

const NGINX_502: &str = "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n\
<body>\r\n<center><h1>502 Bad Gateway</h1></center>\r\n\
<hr><center>nginx/1.25.3 internal-host-7.corp</center>\r\n</body>\r\n</html>\r\n";

#[test]
fn an_html_error_page_is_described_not_forwarded() {
    let err = error_for(502, Some("text/html"), NGINX_502.as_bytes());
    for leaked in ["<html>", "nginx", "internal-host-7.corp", "Bad Gateway"] {
        assert!(!err.contains(leaked), "`{leaked}` reached the error: {err}");
    }
    assert!(err.contains("502"), "the status is kept: {err}");
    assert!(
        err.contains(&format!(
            "unstructured error body (text/html, {} bytes)",
            NGINX_502.len()
        )),
        "{err}"
    );
}

#[test]
fn a_plain_text_refusal_is_described_not_forwarded() {
    let body = "Forbidden: tenant ACME-ORIGIN-TEXT is over quota";
    let err = error_for(403, Some("text/plain; charset=utf-8"), body.as_bytes());
    assert!(!err.contains("ACME-ORIGIN-TEXT"), "{err}");
    assert!(
        err.contains(&format!(
            "unstructured error body (text/plain, {} bytes)",
            body.len()
        )),
        "the media type's essence, parameters dropped: {err}"
    );
}

#[test]
fn an_untyped_or_empty_body_is_described() {
    let raw: &[u8] = b"\xff\xfe raw ORIGIN-BYTES";
    let err = error_for(400, None, raw);
    assert!(!err.contains("ORIGIN-BYTES"), "{err}");
    assert!(
        err.contains(&format!(
            "unstructured error body (no content type, {} bytes)",
            raw.len()
        )),
        "{err}"
    );
    let err = error_for(401, Some("text/plain"), b"");
    assert!(
        err.contains("llm backend returned 401: empty error body"),
        "{err}"
    );
}

/// The header is the origin's text too: a value that is not a media type is
/// not echoed in its place.
#[test]
fn a_content_type_that_is_not_a_media_type_is_not_echoed() {
    let err = error_for(400, Some("<b>ORIGIN-HEADER</b>"), b"nope");
    assert!(!err.contains("ORIGIN-HEADER"), "{err}");
    assert!(
        err.contains("unstructured error body (unrecognized content type, 4 bytes)"),
        "{err}"
    );
}

/// The structured path is unchanged: Ollama's own error shape keeps its reason.
#[test]
fn a_json_error_keeps_its_reason() {
    let err = error_for(
        404,
        Some("application/json"),
        br#"{"error":"model \"m\" not found, try pulling it first"}"#,
    );
    assert!(
        err.contains(r#"llm backend returned 404: model "m" not found, try pulling it first"#),
        "{err}"
    );
}
