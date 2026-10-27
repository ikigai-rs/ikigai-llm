//! What a failure to get an answer becomes (ledger #88, #1546).
//!
//! Every status and every transport failure is typed by ikigai-http's one table
//! (`ikigai_http::status`, 0.3.0), and the variant's class (core 0.1.108's
//! `Error::class`) says who may repeat it. ikigai-throttle's Retry,
//! CircuitBreaker and Failover act on that class:
//!
//! * no connection (refused, DNS failure), a 502, and a 503 without a usable
//!   `Retry-After`: `Unavailable`, transient;
//! * a request that timed out, and a 408/504 answer: `Timeout`, transient;
//! * a 429, and a 503 WITH `Retry-After`: `RetryAfter`, carrying the delay, and
//!   NOT transient (an immediate re-issue goes straight back into the limit);
//! * a 401/403: `Refused` (the server refused the configured key), a 404:
//!   `NotFound`, a 409: `Conflict`;
//! * a 500 and other 5xx, and an answer that came back and is wrong (a malformed
//!   body, any other 4xx): `Endpoint`, class unknown, never retried.
//!
//! Before 0.16.0 this crate kept its own copy of the table, and the copy had
//! drifted: 401, 403 and 404 were `Endpoint`, 429 and 500 were `Unavailable`.
//!
//! No real model is called. Each test runs its own scratch HTTP server on
//! 127.0.0.1, bound to port 0 and kept open for the test's life, and talks to it
//! through a real blocking `ureq` transport, so the transport errors are the
//! ones a native host produces rather than strings written to match.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Error, FailureClass, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig, Registry};

/// A native transport like the CLI's: blocking `ureq`, redirects not followed,
/// and a read timeout short enough for a stalled server to time out in a test.
struct Ureq(ureq::Agent);

impl Ureq {
    fn new() -> Self {
        Ureq(
            ureq::builder()
                .redirects(0)
                .timeout_connect(Duration::from_secs(5))
                .timeout_read(Duration::from_millis(300))
                .build(),
        )
    }
}

#[async_trait]
impl HttpTransport for Ureq {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let mut req = self.0.request(request.method.as_str(), &request.url);
        for (name, value) in &request.headers {
            req = req.set(name, value);
        }
        let outcome = if request.body.is_empty() {
            req.call()
        } else {
            req.send_bytes(&request.body)
        };
        let resp = match outcome {
            Ok(resp) => resp,
            Err(ureq::Error::Status(_, resp)) => resp,
            Err(e) => return Err(e.to_string()),
        };
        let status = resp.status();
        let headers = resp
            .headers_names()
            .into_iter()
            .filter_map(|name| resp.header(&name).map(|v| (name.clone(), v.to_string())))
            .collect();
        let mut body = Vec::new();
        resp.into_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("reading response body: {e}"))?;
        Ok(HttpResponse {
            status,
            headers,
            body: body.into(),
        })
    }
}

/// What the scratch server does with each connection.
#[derive(Clone, Copy)]
enum Behavior {
    /// Answer every request with this status, a content type and a body.
    Answer(u16, &'static str, &'static str),
    /// Answer every request with this status and a `Retry-After` header.
    RetryAfter(u16, &'static str),
    /// Read the request and never answer (held long past the client's timeout).
    Stall,
}

/// Start a scratch server on 127.0.0.1, port 0, and return its `/v1` base URL.
/// The listener is bound HERE and lives in the server thread, so the port is
/// never released and re-picked (field guide 9h).
fn serve(behavior: Behavior) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a scratch port");
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            thread::spawn(move || handle(stream, behavior));
        }
    });
    format!("http://127.0.0.1:{port}/v1")
}

fn handle(mut stream: TcpStream, behavior: Behavior) {
    read_request(&mut stream);
    match behavior {
        Behavior::Answer(status, content_type, body) => {
            let response = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: {content_type}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
        Behavior::RetryAfter(status, value) => {
            let body = r#"{"error":{"message":"slow down"}}"#;
            let response = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                 Retry-After: {value}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
        Behavior::Stall => thread::sleep(Duration::from_secs(5)),
    }
}

/// Read one request: the head, then `Content-Length` bytes of body.
fn read_request(stream: &mut TcpStream) {
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    let head_end = loop {
        let Ok(n) = stream.read(&mut buf) else { return };
        if n == 0 {
            return;
        }
        seen.extend_from_slice(&buf[..n]);
        if let Some(i) = seen.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&seen[..head_end]).to_ascii_lowercase();
    let length: usize = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    while seen.len() < head_end + length {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => seen.extend_from_slice(&buf[..n]),
        }
    }
}

/// A base URL on 127.0.0.1 where nothing listens. The port is bound to port 0
/// and released: this test binary spawns no child process that could inherit
/// the listener (the macOS hazard in field guide 9h), so the port stays closed.
fn closed_base_url() -> String {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    format!("http://127.0.0.1:{port}/v1")
}

fn pinned(base_url: &str) -> OpenAiConfig {
    let mut config = OpenAiConfig::ollama("m");
    config.base_url = base_url.to_string();
    config
}

fn kernel(config: OpenAiConfig) -> Kernel {
    Kernel::new(Arc::new(space(
        Arc::new(Ureq::new()),
        Registry::single(config),
    )))
}

fn issue(kernel: &Kernel, target: &str, args: &[(&str, &str)]) -> Result<String, Error> {
    let mut request = Request::new(Verb::Source, Iri::parse(target).unwrap());
    for (name, value) in args {
        request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
    }
    futures::executor::block_on(kernel.issue(request, &Capability::root()))
        .map(|repr| String::from_utf8_lossy(&repr.bytes).into_owned())
}

/// `urn:llm:ollama:ask` with the model named, so no fallback probe runs.
fn ask(base_url: &str) -> Error {
    issue(
        &kernel(pinned(base_url)),
        "urn:llm:ollama:ask",
        &[("prompt", "hi"), ("model", "m")],
    )
    .expect_err("the backend did not answer usefully")
}

#[test]
fn a_refused_connection_is_unavailable() {
    let err = ask(&closed_base_url());
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    assert!(err.is_transient());
}

#[test]
fn a_dns_failure_is_unavailable() {
    // `.invalid` never resolves (RFC 6761 section 6.4).
    let err = ask("http://ikigai-llm-backend.invalid:11434/v1");
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    assert!(err.is_transient());
}

#[test]
fn a_request_that_times_out_is_a_timeout() {
    let err = ask(&serve(Behavior::Stall));
    assert!(matches!(err, Error::Timeout(_)), "{err:?}");
    assert!(err.is_transient());
}

#[test]
fn a_502_or_a_503_without_a_delay_is_unavailable() {
    for status in [502, 503] {
        let err = ask(&serve(Behavior::Answer(status, "text/plain", "busy")));
        assert!(matches!(err, Error::Unavailable(_)), "{status}: {err:?}");
        assert!(err.is_transient(), "{status}");
        assert_eq!(err.class(), FailureClass::Transient, "{status}");
        // What the error says is unchanged: the status, and the body described.
        assert!(
            err.to_string()
                .contains(&format!("llm backend returned {status}")),
            "{err}"
        );
    }
}

/// A 429 is a rate limit, `RetryAfter`, with the delay its header states; it was
/// `Unavailable` (transient) before 0.16.0, so a Retry overlay re-issued straight
/// into the limit. A 503 that states a delay is `RetryAfter` too.
#[test]
fn a_429_or_a_503_with_retry_after_is_retry_after_with_its_delay() {
    for status in [429, 503] {
        let err = ask(&serve(Behavior::RetryAfter(status, "7")));
        assert!(
            matches!(err, Error::RetryAfter { delay: Some(d), .. } if d == Duration::from_secs(7)),
            "{status}: {err:?}"
        );
        assert_eq!(err.class(), FailureClass::RetryAfter, "{status}");
        assert!(!err.is_transient(), "{status}");
        assert_eq!(err.retry_after_millis(), Some(7_000), "{status}");
        assert!(
            err.to_string()
                .contains(&format!("llm backend returned {status}: slow down")),
            "{err}"
        );
    }
    // A 429 with no header is still a rate limit: "not now", without a time.
    let err = ask(&serve(Behavior::Answer(429, "text/plain", "busy")));
    assert!(
        matches!(err, Error::RetryAfter { delay: None, .. }),
        "{err:?}"
    );
}

/// A 500 reports the server's own failure and is no promise that asking again
/// helps: `Endpoint`, class unknown. It was `Unavailable` before 0.16.0.
#[test]
fn a_500_is_an_untyped_endpoint_error_never_retried() {
    for status in [500, 505] {
        let err = ask(&serve(Behavior::Answer(status, "text/plain", "broken")));
        assert!(matches!(err, Error::Endpoint(_)), "{status}: {err:?}");
        assert!(!err.is_transient(), "{status}");
        assert_eq!(err.class(), FailureClass::Unknown, "{status}");
    }
}

/// The server refused the configured key: `Refused`, not the capability floor's
/// `Denied` and not an opaque `Endpoint` (as it was before 0.16.0).
#[test]
fn a_401_or_403_is_refused() {
    for status in [401, 403] {
        let err = ask(&serve(Behavior::Answer(
            status,
            "application/json",
            r#"{"error":"bad key"}"#,
        )));
        assert!(matches!(err, Error::Refused(_)), "{status}: {err:?}");
        assert_eq!(err.class(), FailureClass::Refused, "{status}");
        assert!(err.to_string().contains("bad key"), "{err}");
    }
}

/// A named model the server does not have: `NotFound` (it was `Endpoint`).
#[test]
fn a_404_is_not_found() {
    let err = ask(&serve(Behavior::Answer(
        404,
        "application/json",
        r#"{"error":"model not found"}"#,
    )));
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    assert_eq!(err.class(), FailureClass::Permanent);
}

#[test]
fn a_gateway_or_request_timeout_answer_is_a_timeout() {
    for status in [504, 408] {
        let err = ask(&serve(Behavior::Answer(status, "text/plain", "slow")));
        assert!(matches!(err, Error::Timeout(_)), "{status}: {err:?}");
        assert!(err.is_transient(), "{status}");
    }
}

#[test]
fn any_other_4xx_stays_an_untyped_endpoint_error() {
    for status in [400, 422] {
        let err = ask(&serve(Behavior::Answer(
            status,
            "application/json",
            r#"{"error":"no"}"#,
        )));
        assert!(matches!(err, Error::Endpoint(_)), "{status}: {err:?}");
        assert!(!err.is_transient(), "{status}");
        assert_eq!(err.class(), FailureClass::Unknown, "{status}");
    }
}

#[test]
fn a_malformed_answer_stays_a_permanent_endpoint_error() {
    let err = ask(&serve(Behavior::Answer(200, "text/plain", "not json")));
    assert!(matches!(err, Error::Endpoint(_)), "{err:?}");
    assert!(!err.is_transient());

    let err = ask(&serve(Behavior::Answer(200, "application/json", "{}")));
    assert!(matches!(err, Error::Endpoint(_)), "{err:?}");
}

/// The facade issues the backend as a sub-request; the type must survive it,
/// or an overlay in front of `urn:llm:ask` would still see nothing transient.
#[test]
fn the_facade_keeps_the_type() {
    let err = issue(
        &kernel(pinned(&closed_base_url())),
        "urn:llm:ask",
        &[("prompt", "hi"), ("model", "m")],
    )
    .expect_err("down");
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");

    let err = issue(
        &kernel(pinned(&serve(Behavior::RetryAfter(429, "3")))),
        "urn:llm:ask",
        &[("prompt", "hi"), ("model", "m")],
    )
    .expect_err("limited");
    assert_eq!(err.retry_after_millis(), Some(3_000), "{err:?}");
}

/// A discovering provider asks the server which model it serves before it can
/// ask anything else; that failure is wrapped with context, and keeps its type.
#[test]
fn a_failed_model_discovery_keeps_the_type() {
    let k = kernel(OpenAiConfig::discovering("srv", closed_base_url()));
    let err = issue(&k, "urn:llm:srv:ask", &[("prompt", "hi")]).expect_err("down");
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    assert!(
        err.to_string().contains("could not discover a model"),
        "{err}"
    );

    let k = kernel(OpenAiConfig::discovering(
        "srv",
        serve(Behavior::Answer(503, "text/plain", "busy")),
    ));
    let err = issue(&k, "urn:llm:srv:ask", &[("prompt", "hi")]).expect_err("busy");
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");

    // A listing that came back malformed is not transient.
    let k = kernel(OpenAiConfig::discovering(
        "srv",
        serve(Behavior::Answer(200, "text/plain", "not json")),
    ));
    let err = issue(&k, "urn:llm:srv:ask", &[("prompt", "hi")]).expect_err("garbage");
    assert!(matches!(err, Error::Endpoint(_)), "{err:?}");
}

/// The discovery failure is reworded with context, and every type the table can
/// return survives the rewording, a `RetryAfter`'s delay included. Before 0.16.0 a
/// 404 or a 409 became `Endpoint` here, and so did everything the copy did not name.
#[test]
fn rewording_a_discovery_failure_keeps_every_type() {
    let discover = |behavior| {
        let k = kernel(OpenAiConfig::discovering("srv", serve(behavior)));
        issue(&k, "urn:llm:srv:ask", &[("prompt", "hi")]).expect_err("failed")
    };
    let json = "application/json";
    let cases: [(Behavior, &str); 6] = [
        (Behavior::Answer(401, json, "{}"), "refused"),
        (Behavior::Answer(404, json, "{}"), "not-found"),
        (Behavior::Answer(409, json, "{}"), "conflict"),
        (Behavior::Answer(413, json, "{}"), "over-budget"),
        (Behavior::Answer(501, json, "{}"), "unsupported"),
        (Behavior::RetryAfter(429, "5"), "retry-after"),
    ];
    for (behavior, kind) in cases {
        let err = discover(behavior);
        assert_eq!(err.kind(), kind, "{err:?}");
        assert!(
            err.to_string().contains("could not discover a model"),
            "the context is added: {err}"
        );
    }
    let err = discover(Behavior::RetryAfter(429, "5"));
    assert_eq!(
        err.retry_after_millis(),
        Some(5_000),
        "the delay survives: {err:?}"
    );
}

#[test]
fn the_installed_listing_is_typed_the_same_way() {
    let mut config = OpenAiConfig::discovering("srv", closed_base_url());
    config.default_model = Some("m".into());
    let err = issue(&kernel(config), "urn:llm:srv:installed", &[]).expect_err("down");
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");

    let mut config =
        OpenAiConfig::discovering("srv", serve(Behavior::Answer(502, "text/html", "<p>")));
    config.default_model = Some("m".into());
    let err = issue(&kernel(config), "urn:llm:srv:installed", &[]).expect_err("down");
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
}

/// Liveness is unchanged: a server that cannot be reached is `false`, not an
/// error.
#[test]
fn up_still_answers_false_for_a_down_server() {
    let k = kernel(pinned(&closed_base_url()));
    assert_eq!(issue(&k, "urn:llm:ollama:up", &[]).unwrap(), "false");
}
