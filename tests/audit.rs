//! The reproductions from the unled audit of `a736b31` (ledger #884), deduplicated
//! across its two auditors and ported to the published API. Every test here failed
//! on `a736b31` because of the defect its comment names. (`Claude rN` and `Hermes
//! bugN` name the auditor's own reproduction, kept in ikigai-devtools under
//! `claude/research/audit-ikigai-llm-2026-10-07/`.)
//!
//! No real model is called: [`Stub`] is an in-process [`HttpTransport`] that
//! answers the way the named server does (Ollama, an OpenAI-shaped API) and logs
//! every request it is sent, so a test can prove what was — and was not — asked.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, ContentId, Error, Expiry, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig, Registry};

// ---- a scripted stub transport --------------------------------------------------

type Handler = dyn Fn(&HttpRequest) -> Result<HttpResponse, String> + Send + Sync;

struct Stub {
    handler: Box<Handler>,
    log: Mutex<Vec<HttpRequest>>,
}

impl Stub {
    fn new(
        handler: impl Fn(&HttpRequest) -> Result<HttpResponse, String> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Stub {
            handler: Box::new(handler),
            log: Mutex::new(Vec::new()),
        })
    }

    fn sent(&self) -> Vec<HttpRequest> {
        self.log.lock().unwrap().clone()
    }

    fn urls(&self) -> Vec<String> {
        self.sent().into_iter().map(|r| r.url).collect()
    }

    fn bodies(&self) -> Vec<String> {
        self.sent()
            .into_iter()
            .map(|r| String::from_utf8_lossy(&r.body).into_owned())
            .collect()
    }
}

#[async_trait]
impl HttpTransport for Stub {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let out = (self.handler)(&request);
        self.log.lock().unwrap().push(request);
        out
    }
}

fn reply(status: u16, body: &str) -> Result<HttpResponse, String> {
    Ok(HttpResponse {
        status,
        headers: vec![],
        body: body.as_bytes().to_vec(),
    })
}

const CANNED: &str = r#"{"model":"m","choices":[{"message":{"role":"assistant","content":"Hello there!"},"finish_reason":"stop"}]}"#;

fn ok_stub() -> Arc<Stub> {
    Stub::new(|_| reply(200, CANNED))
}

fn kernel(stub: Arc<Stub>, registry: Registry) -> Kernel {
    let transport: Arc<dyn HttpTransport> = stub;
    Kernel::new(Arc::new(space(transport, registry)))
}

fn registry(json: &str) -> Registry {
    Registry::from_json(json).unwrap()
}

fn req(iri: &str, args: &[(&str, &str)]) -> Request {
    let mut r = Request::new(Verb::Source, Iri::parse(iri).unwrap());
    for (k, v) in args {
        r = r.with_arg(*k, ArgRef::Inline(v.as_bytes().to_vec()));
    }
    r
}

fn run(k: &Kernel, r: Request, cap: &Capability) -> Result<ikigai_core::Representation, Error> {
    futures::executor::block_on(k.issue(r, cap))
}

fn root(k: &Kernel, r: Request) -> Result<ikigai_core::Representation, Error> {
    run(k, r, &Capability::root())
}

fn text(rep: &ikigai_core::Representation) -> String {
    String::from_utf8_lossy(&rep.bytes).into_owned()
}

fn scoped(scopes: &[&str]) -> Capability {
    Capability::scoped(scopes.iter().map(|s| s.to_string()))
}

// ---- serious 1: the network gates ignore the port --------------------------------

const TWO_LOCAL: &str = r#"{ "default": "ollama", "providers": {
    "ollama": { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b" },
    "mlx":    { "base_url": "http://localhost:8000/v1",  "model": "qwen" } } }"#;

/// A grant scoped to `localhost:8000` (the MLX server) must not reach Ollama on
/// `localhost:11434`. Both auditors (Claude r1a, Hermes bug5).
#[test]
fn a_grant_on_one_port_does_not_reach_another_port() {
    let cap = scoped(&["urn:cap:net:localhost:8000"]);
    assert!(!ikigai_http::net_allows_port(
        &cap,
        "localhost",
        Some(11434),
        "/v1/chat/completions"
    ));
    let stub = ok_stub();
    let k = kernel(stub.clone(), registry(TWO_LOCAL));
    let out = run(&k, req("urn:llm:ollama:ask", &[("prompt", "hi")]), &cap);
    assert!(
        matches!(out, Err(Error::Denied(_))),
        "a grant on localhost:8000 reached localhost:11434: {:?}, sent {:?}",
        out.map(|r| text(&r)),
        stub.urls()
    );
    assert!(
        stub.urls().is_empty(),
        "nothing was sent: {:?}",
        stub.urls()
    );
    // ...while the port it names still connects.
    assert!(run(&k, req("urn:llm:mlx:ask", &[("prompt", "hi")]), &cap).is_ok());
}

/// The mirror image: a deny on ONE port must not refuse a provider on another
/// port that the broader allow covers (Claude r1b).
#[test]
fn a_deny_on_one_port_does_not_refuse_another_port() {
    let cap = scoped(&["urn:cap:net:localhost", "urn:cap:net:-localhost:11434"]);
    let k = kernel(ok_stub(), registry(TWO_LOCAL));
    let out = run(&k, req("urn:llm:mlx:ask", &[("prompt", "hi")]), &cap);
    assert!(
        out.is_ok(),
        "a deny on :11434 refused :8000: {:?}",
        out.err()
    );
    let denied = run(&k, req("urn:llm:ollama:ask", &[("prompt", "hi")]), &cap);
    assert!(matches!(denied, Err(Error::Denied(_))), "{denied:?}");
}

/// The native-API gates (`/api/show`, `/api/tags`) honor the port too: a grant
/// on another port of the same host must not let discovery or the installed
/// listing reach Ollama.
#[test]
fn discovery_and_listing_honor_the_port() {
    let stub = Stub::new(|r| {
        if r.url.ends_with("/api/show") {
            reply(200, r#"{"capabilities":["completion","tools"]}"#)
        } else if r.url.ends_with("/api/tags") {
            reply(200, r#"{"models":[{"name":"m","size":1}]}"#)
        } else {
            reply(200, r#"{"data":[{"id":"m"}]}"#)
        }
    });
    let k = kernel(
        stub.clone(),
        registry(
            r#"{ "default": "o", "providers": { "o": {
            "base_url": "http://localhost:11434/v1", "model": "m",
            "caps": { "vendor": "ollama", "cost": "local" } } } }"#,
        ),
    );
    let elsewhere = scoped(&["urn:cap:net:localhost:8000"]);
    let _ = run(&k, req("urn:llm:models", &[]), &elsewhere);
    let installed = run(&k, req("urn:llm:o:installed", &[]), &elsewhere);
    assert!(matches!(installed, Err(Error::Denied(_))), "{installed:?}");
    assert!(
        stub.urls().is_empty(),
        "a grant on :8000 reached :11434: {:?}",
        stub.urls()
    );
}

// ---- serious 2: a pinned ollama provider's live answers were cached forever -----

const OLLAMA_AND_PREMIUM: &str = r#"{ "default": "o", "providers": {
    "o": { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
           "caps": { "vendor": "ollama", "cost": "local" } },
    "p": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "k",
           "caps": { "vendor": "openai", "cost": "premium", "tools": true } } } }"#;

fn show_stub(up: Arc<AtomicBool>) -> Arc<Stub> {
    Stub::new(move |r| {
        if r.url.ends_with("/api/show") {
            if up.load(Ordering::SeqCst) {
                reply(200, r#"{"capabilities":["completion","tools"]}"#)
            } else {
                Err("connection refused".to_string())
            }
        } else {
            reply(404, "{}")
        }
    })
}

/// `urn:llm:select` for a pinned `vendor: "ollama"` provider probes `/api/show`
/// on every resolve, so its answer is a live fact. A selection taken while Ollama
/// was briefly down must not be served after it comes back (Claude r2, Hermes
/// bug2).
#[test]
fn a_select_taken_while_ollama_was_down_is_not_served_after_it_returns() {
    let up = Arc::new(AtomicBool::new(false));
    let k = kernel(show_stub(up.clone()), registry(OLLAMA_AND_PREMIUM));
    let r = req("urn:llm:select", &[("needs", "tools")]);

    let first = root(&k, r.clone()).unwrap();
    assert_eq!(
        text(&first),
        "urn:llm:p:ask",
        "while /api/show is down only p declares tools"
    );
    assert_ne!(
        first.expiry,
        Expiry::Never,
        "a network-fed answer is not permanent"
    );

    up.store(true, Ordering::SeqCst);
    let second = root(&k, r).unwrap();
    assert_eq!(
        text(&second),
        "urn:llm:o:ask",
        "the outage's selection was served from cache"
    );
}

// ---- serious 3: the bearer key never reached listing, liveness, discovery -------

const REMOTE_PINNED: &str = r#"{ "default": "remote", "providers": {
    "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "sk-test" } } }"#;
const REMOTE_DISCOVERING: &str = r#"{ "default": "remote", "providers": {
    "remote": { "base_url": "https://api.example.com/v1", "api_key": "sk-test" } } }"#;

fn authed(r: &HttpRequest) -> bool {
    r.headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("authorization") && v == "Bearer sk-test")
}

/// An OpenAI-shaped API: every route, `/models` included, answers 401 without
/// the bearer token (OpenAI's `GET /v1/models` does exactly this).
fn authed_api() -> Arc<Stub> {
    Stub::new(|r| {
        if !authed(r) {
            return reply(
                401,
                r#"{"error":{"message":"You didn't provide an API key."}}"#,
            );
        }
        if r.url.ends_with("/models") {
            reply(200, r#"{"object":"list","data":[{"id":"gpt-4o"}]}"#)
        } else {
            reply(200, CANNED)
        }
    })
}

/// `:up` reported `false` for a live, correctly keyed provider (Claude r3a).
#[test]
fn up_is_true_for_a_live_keyed_provider() {
    let k = kernel(authed_api(), registry(REMOTE_PINNED));
    assert!(root(&k, req("urn:llm:remote:ask", &[("prompt", "hi")])).is_ok());
    let up = root(&k, req("urn:llm:remote:up", &[])).unwrap();
    assert_eq!(text(&up), "true", "a live, keyed provider reported down");
}

/// A provider that names only its server AND needs a key could not answer at
/// all: discovery listed `/models` without the key (Claude r3b).
#[test]
fn a_keyed_discovering_provider_can_answer() {
    let k = kernel(authed_api(), registry(REMOTE_DISCOVERING));
    let out = root(&k, req("urn:llm:remote:ask", &[("prompt", "hi")]));
    assert!(
        out.is_ok(),
        "keyed discovering provider failed: {:?}",
        out.err()
    );
}

/// Every request a keyed provider makes carries the key: chat, the OpenAI-compat
/// listing, liveness, and Ollama's native `/api/tags` and `/api/show`.
#[test]
fn every_request_a_keyed_provider_makes_carries_the_key() {
    let stub = Stub::new(|r| {
        if r.url.ends_with("/api/show") {
            reply(200, r#"{"capabilities":["completion"]}"#)
        } else if r.url.ends_with("/api/tags") {
            reply(200, r#"{"models":[{"name":"m","size":1}]}"#)
        } else if r.url.ends_with("/models") {
            reply(200, r#"{"data":[{"id":"m"}]}"#)
        } else {
            reply(200, CANNED)
        }
    });
    let k = kernel(
        stub.clone(),
        registry(
            r#"{ "default": "o", "providers": { "o": {
            "base_url": "https://ollama.example.com/v1", "model": "m", "api_key": "sk-test",
            "caps": { "vendor": "ollama", "cost": "local" } } } }"#,
        ),
    );
    for r in [
        req("urn:llm:o:ask", &[("prompt", "hi")]),
        req("urn:llm:o:up", &[]),
        req("urn:llm:o:installed", &[]),
        req("urn:llm:models", &[]),
        req("urn:llm:select", &[("needs", "cost=local")]),
    ] {
        root(&k, r).unwrap();
    }
    let sent = stub.sent();
    for path in ["/chat/completions", "/v1/models", "/api/tags", "/api/show"] {
        assert!(
            sent.iter().any(|r| r.url.ends_with(path)),
            "{path} was exercised"
        );
    }
    for r in &sent {
        assert!(authed(r), "{} was sent without the key", r.url);
    }
}

// ---- serious 6: credentials in base_url were served to a capability-less caller --

const PROXIED: &str = r#"{ "default": "proxied", "providers": { "proxied": {
    "base_url": "https://alice:s3cret@llm.example.com/v1", "model": "m", "api_key": "sk-x" } } }"#;

/// `urn:llm:config` (no capability) and `urn:llm:models` echoed `base_url`
/// verbatim, password included (Claude r6).
#[test]
fn no_face_serves_userinfo_credentials() {
    let k = kernel(Stub::new(|_| reply(500, "{}")), registry(PROXIED));
    let nobody = scoped(&[]);
    for (iri, args) in [
        ("urn:llm:config", vec![]),
        ("urn:llm:models", vec![]),
        ("urn:llm:models", vec![("as", "text/turtle")]),
    ] {
        let out = run(&k, req(iri, &args), &nobody).unwrap();
        assert!(
            !text(&out).contains("s3cret") && !text(&out).contains("alice"),
            "{iri} {args:?} served a credential to a caller with no grants: {}",
            text(&out)
        );
    }
    // The location is still reported, without its userinfo.
    let config = text(&run(&k, req("urn:llm:config", &[]), &nobody).unwrap());
    assert!(config.contains("https://llm.example.com/v1"), "{config}");
}

/// The composed errors redact it too: a discovery failure names the base_url,
/// and a transport error (ureq's carries the URL) is scrubbed.
#[test]
fn no_error_carries_userinfo_credentials() {
    let stub = Stub::new(|r| Err(format!("{}: Connection refused", r.url)));
    let k = kernel(
        stub,
        registry(
            r#"{ "default": "d", "providers": {
            "d": { "base_url": "https://alice:s3cret@llm.example.com/v1" },
            "p": { "base_url": "https://alice:s3cret@llm.example.com/v1", "model": "m" } } }"#,
        ),
    );
    for iri in ["urn:llm:d:ask", "urn:llm:d:model", "urn:llm:p:ask"] {
        let err = root(&k, req(iri, &[("prompt", "hi")])).unwrap_err();
        let shown = format!("{err} / {err:?}");
        assert!(
            !shown.contains("s3cret"),
            "{iri}: the error carries the password: {shown}"
        );
    }
}

// ---- the declaration and the disclosure ------------------------------------------

const LOCAL_AND_REMOTE: &str = r#"{ "default": "fast", "providers": {
    "fast": { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
              "caps": { "cost": "local", "vendor": "mock", "context": 32768 } },
    "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o",
                "api_key": "sk-SECRET", "caps": { "cost": "premium", "vendor": "openai",
                "context": 131072, "tools": true } } } }"#;

/// A localhost-only capability was told which REMOTE backend satisfies `tools`
/// — its IRI, model and vendor — though it holds no grant to reach it
/// (Hermes bug9). Selection now reasons only over what the caller can reach.
#[test]
fn select_never_reports_a_backend_the_caller_cannot_reach() {
    let stub = ok_stub();
    let k = kernel(stub.clone(), registry(LOCAL_AND_REMOTE));
    let local_only = scoped(&["urn:cap:net:localhost"]);
    let out = run(
        &k,
        req(
            "urn:llm:select",
            &[("needs", "tools"), ("as", "application/json")],
        ),
        &local_only,
    );
    match out {
        Ok(rep) => panic!("disclosed {}", text(&rep)),
        Err(e) => {
            let shown = format!("{e} / {e:?}");
            assert!(
                !shown.contains("gpt-4o") && !shown.contains("api.example.com"),
                "{shown}"
            );
        }
    }
    // The facade routes the same way and sends nothing.
    let ask = run(
        &k,
        req("urn:llm:ask", &[("prompt", "hi"), ("needs", "tools")]),
        &local_only,
    );
    assert!(ask.is_err(), "{:?}", ask.map(|r| text(&r)));
    assert!(stub.urls().is_empty(), "{:?}", stub.urls());
    // A requirement both satisfy goes to the one the caller can reach, even
    // where the unreachable one would otherwise win the tie-break.
    let reg = registry(
        r#"{ "default": "fast", "providers": {
        "a_remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o",
                      "caps": { "cost": "local", "vendor": "openai", "context": 4096 } },
        "fast": { "base_url": "http://localhost:11434/v1", "model": "m",
                  "caps": { "cost": "local", "vendor": "mock", "context": 32768 } } } }"#,
    );
    let k = kernel(ok_stub(), reg);
    assert_eq!(
        text(&root(&k, req("urn:llm:select", &[("needs", "cost=local")])).unwrap()),
        "urn:llm:a_remote:ask",
        "root reaches both: the smaller context wins"
    );
    assert_eq!(
        text(
            &run(
                &k,
                req("urn:llm:select", &[("needs", "cost=local")]),
                &local_only
            )
            .unwrap()
        ),
        "urn:llm:fast:ask"
    );
}

/// `urn:llm:select` reaches the network (discovery) and answers only with a
/// backend the caller can reach, so it declares the net capability (Hermes
/// bug6, the `select` half).
#[test]
fn select_declares_the_net_capability() {
    let k = kernel(ok_stub(), registry(LOCAL_AND_REMOTE));
    let described = k.describe(&Iri::parse("urn:llm:select").unwrap()).unwrap();
    let requires: Vec<String> = described
        .action_specs()
        .into_iter()
        .flat_map(|s| s.requires)
        .collect();
    assert!(
        requires.iter().any(|c| c == "urn:cap:net:*"),
        "{requires:?}"
    );
    let denied = run(
        &k,
        req("urn:llm:select", &[("needs", "cost=local")]),
        &scoped(&[]),
    );
    assert!(matches!(denied, Err(Error::Denied(_))), "{denied:?}");
}

// ---- minors ------------------------------------------------------------------------

/// `ctx>=18014398509481985k` is 2^64 + 1024 tokens: the unchecked multiply
/// panicked in debug and wrapped to 1024 in release, SELECTING a 128k backend
/// (Claude r4).
#[test]
fn an_oversized_context_requirement_is_refused_not_wrapped() {
    let k = kernel(
        ok_stub(),
        registry(
            r#"{ "default": "fast", "providers": { "fast": {
            "base_url": "http://localhost:11434/v1", "model": "m",
            "caps": { "context": 131072, "cost": "local" } } } }"#,
        ),
    );
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        root(
            &k,
            req("urn:llm:select", &[("needs", "ctx>=18014398509481985k")]),
        )
    }));
    match outcome {
        Err(_) => panic!("urn:llm:select panicked on a caller-supplied needs= value"),
        Ok(Ok(rep)) => panic!("2^64+1024 tokens selected {}", text(&rep)),
        Ok(Err(e)) => assert!(
            matches!(e, Error::InvalidArgument { .. }),
            "an unrepresentable count is a grammar error: {e:?}"
        ),
    }
}

/// A raw line break inside a `"…"` literal is not Turtle, and a discovering
/// provider's model id comes from the SERVER (Claude r5, Hermes bug11).
#[test]
fn the_turtle_inventory_parses_whatever_the_server_reports() {
    let stub = Stub::new(|r| {
        if r.url.ends_with("/models") {
            reply(
                200,
                r#"{"data":[{"id":"qwen3\nlatest\r","modality":"te\nxt"}]}"#,
            )
        } else {
            reply(404, "{}")
        }
    });
    let k = kernel(
        stub,
        registry(
            r#"{ "default": "rapid", "providers": { "rapid": { "base_url": "http://localhost:8000/v1" } } }"#,
        ),
    );
    let ttl = root(&k, req("urn:llm:models", &[("as", "text/turtle")])).unwrap();
    let parsed = ikigai_conformance::rdf::parse("text/turtle", &ttl.bytes);
    assert!(parsed.is_ok(), "invalid Turtle {parsed:?}:\n{}", text(&ttl));
    let triples = parsed.unwrap();
    assert!(
        format!("{triples:?}").contains(r"qwen3\nlatest\r"),
        "the value survives the round trip: {triples:?}"
    );
}

/// A `default` that names no provider loaded; every unrouted ask then failed and
/// the inventory asserted a route to nothing (Claude r7, Hermes bug8).
#[test]
fn a_default_naming_no_provider_is_refused_at_load() {
    let err = Registry::from_json(
        r#"{ "default": "nope", "providers": {
        "fast": { "base_url": "http://localhost:11434/v1", "model": "m" } } }"#,
    )
    .expect_err("a default naming no provider must not load");
    let msg = format!("{err:?}");
    assert!(msg.contains("nope") && msg.contains("fast"), "{msg}");
}

/// After the default-model fallback the span was labeled with the model that
/// 404'd, not the one that answered (Claude r8).
#[test]
fn the_trace_names_the_model_that_answered_after_the_fallback() {
    struct Rec(Mutex<Vec<ikigai_core::TraceEvent>>);
    impl ikigai_core::Tracer for Rec {
        fn record(&self, event: ikigai_core::TraceEvent) {
            self.0.lock().unwrap().push(event);
        }
    }
    let stub = Stub::new(|r| {
        let body = String::from_utf8_lossy(&r.body).into_owned();
        if r.url.ends_with("/chat/completions") {
            if body.contains("\"ghost\"") {
                reply(404, r#"{"error":{"message":"model not found"}}"#)
            } else {
                reply(
                    200,
                    r#"{"model":"small:3b","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}"#,
                )
            }
        } else if r.url.ends_with("/api/tags") {
            reply(200, r#"{"models":[{"name":"small:3b","size":2000000000}]}"#)
        } else if r.url.ends_with("/api/show") {
            reply(200, r#"{"capabilities":["completion"]}"#)
        } else {
            reply(404, "{}")
        }
    });
    let k = kernel(stub, Registry::single(OpenAiConfig::ollama("ghost")));
    let rec = Arc::new(Rec(Mutex::new(Vec::new())));
    let out = futures::executor::block_on(k.issue_traced(
        req("urn:llm:ollama:ask", &[("prompt", "hi")]),
        &Capability::root(),
        rec.clone(),
    ))
    .unwrap();
    assert_eq!(text(&out), "hi", "the fallback answered");
    let events = rec.0.lock().unwrap();
    let span = events
        .iter()
        .find(|e| e.target == "urn:llm:ollama:ask")
        .unwrap();
    let models: Vec<&String> = span
        .notes
        .iter()
        .filter(|(k, _)| k == "model")
        .map(|(_, v)| v)
        .collect();
    assert_eq!(models, ["small:3b"], "the span names the answering model");
}

/// The facade tested `if let Ok(needs) = …`, so a `needs=` it could not read (a
/// reference to a policy resource, an interned value) was dropped and the ask
/// went to the DEFAULT — here the very vendor the expression excludes (Claude
/// r9). `provider=` had the same shape.
#[test]
fn an_unreadable_routing_input_is_refused_not_ignored() {
    let stub = ok_stub();
    let k = kernel(
        stub.clone(),
        registry(
            r#"{ "default": "posh", "providers": {
            "posh":  { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "k",
                       "caps": { "vendor": "openai", "cost": "premium" } },
            "local": { "base_url": "http://localhost:11434/v1", "model": "m",
                       "caps": { "vendor": "mlx", "cost": "local" } } } }"#,
        ),
    );
    for name in ["needs", "provider"] {
        for (label, value) in [
            (
                "reference",
                ArgRef::Reference(Iri::parse("urn:policy:no-openai").unwrap()),
            ),
            ("content", ArgRef::Content(ContentId::of(b"vendor!=openai"))),
        ] {
            let r = req("urn:llm:ask", &[("prompt", "hi")]).with_arg(name, value);
            let out = root(&k, r);
            assert!(
                matches!(out, Err(Error::InvalidArgument { .. })),
                "{name}= as {label} was not refused: {:?}",
                out.map(|r| text(&r))
            );
        }
    }
    assert!(stub.urls().is_empty(), "routed anyway: {:?}", stub.urls());
}

/// A JSON error body without `error.message` fell through verbatim; FastAPI's
/// 422 carries the offending request value, so the prompt reached the error
/// (Claude r10).
#[test]
fn a_structured_error_body_does_not_carry_the_prompt_into_the_error() {
    const PROMPT: &str = "SECRET-PROMPT-never-echoed";
    let stub = Stub::new(|r| {
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        let body = serde_json::json!({ "detail": [ {
            "type": "string_too_long", "loc": ["body", "messages", 0, "content"],
            "msg": "String should have at most 8 characters",
            "input": v["messages"][0]["content"] } ] });
        reply(422, &body.to_string())
    });
    let k = kernel(stub, Registry::single(OpenAiConfig::ollama("m")));
    let err = root(
        &k,
        req("urn:llm:ollama:ask", &[("prompt", PROMPT), ("model", "m")]),
    )
    .unwrap_err()
    .to_string();
    assert!(!err.contains(PROMPT), "the prompt is in the error: {err}");
    assert!(
        err.contains("String should have at most 8 characters"),
        "the reason is kept: {err}"
    );
    // A plain-text body that echoes the prompt is scrubbed too.
    let stub = Stub::new(|r| {
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        reply(
            400,
            &format!(
                "bad request: {}",
                v["messages"][0]["content"].as_str().unwrap()
            ),
        )
    });
    let k = kernel(stub, Registry::single(OpenAiConfig::ollama("m")));
    let err = root(
        &k,
        req("urn:llm:ollama:ask", &[("prompt", PROMPT), ("model", "m")]),
    )
    .unwrap_err()
    .to_string();
    assert!(!err.contains(PROMPT), "the prompt is in the error: {err}");
}

/// An additive `modality` annotation discarded every discovered modality on a
/// provider that declared none (Claude r11).
#[test]
fn an_added_modality_keeps_the_discovered_ones() {
    let stub = Stub::new(|r| {
        if r.url.ends_with("/api/show") {
            reply(200, r#"{"capabilities":["completion"]}"#)
        } else {
            reply(404, "{}")
        }
    });
    let mut reg = registry(
        r#"{ "default": "o", "providers": { "o": {
        "base_url": "http://localhost:11434/v1", "model": "m",
        "caps": { "vendor": "ollama", "cost": "local" } } } }"#,
    );
    let before = kernel(stub.clone(), reg.clone());
    assert_eq!(
        text(&root(&before, req("urn:llm:select", &[("needs", "text")])).unwrap()),
        "urn:llm:o:ask"
    );
    assert!(reg
        .apply_annotations(&[("o", "ik:modality", "vision")])
        .is_empty());
    let after = kernel(stub, reg);
    let out = root(&after, req("urn:llm:select", &[("needs", "text, vision")]));
    assert_eq!(
        out.as_ref().map(text).ok().as_deref(),
        Some("urn:llm:o:ask"),
        "adding `vision` dropped the discovered `text`: {:?}",
        out.err()
    );
}

/// Typed inputs that do not parse were silently dropped (`temperature=abc`
/// sent no temperature), a non-finite temperature was sent as JSON `null`, and
/// `max_tokens=0` (declared `xsd:positiveInteger`) was sent as 0 (Hermes bug3,
/// bug7, and its suspected `max_tokens=0`).
#[test]
fn a_typed_input_that_does_not_parse_is_refused_not_dropped() {
    for (name, value) in [
        ("temperature", "abc"),
        ("temperature", "NaN"),
        ("temperature", "inf"),
        ("temperature", "-inf"),
        ("max_tokens", "abc"),
        ("max_tokens", "0"),
        ("max_tokens", "-3"),
    ] {
        let stub = ok_stub();
        let k = kernel(stub.clone(), Registry::single(OpenAiConfig::ollama("m")));
        let out = root(
            &k,
            req("urn:llm:ollama:ask", &[("prompt", "hi"), (name, value)]),
        );
        assert!(
            matches!(&out, Err(Error::InvalidArgument { name: n, .. }) if n == name),
            "{name}={value} was not refused: {:?}; sent {:?}",
            out.map(|r| text(&r)),
            stub.bodies()
        );
        assert!(stub.urls().is_empty(), "{name}={value}: a request was sent");
    }
    // The valid forms still reach the payload.
    let stub = ok_stub();
    let k = kernel(stub.clone(), Registry::single(OpenAiConfig::ollama("m")));
    root(
        &k,
        req(
            "urn:llm:ollama:ask",
            &[
                ("prompt", "hi"),
                ("temperature", "0.2"),
                ("max_tokens", "64"),
            ],
        ),
    )
    .unwrap();
    let body: serde_json::Value = serde_json::from_str(&stub.bodies()[0]).unwrap();
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["max_tokens"], 64);
}

/// The 404 default-model fallback on the OpenAI-compat path retried with the
/// FIRST listed model, though that listing says nothing about what can chat —
/// here an embedder (Hermes bug12). A substitution needs evidence.
#[test]
fn the_fallback_never_substitutes_a_model_not_known_to_chat() {
    let stub = Stub::new(|r| {
        if r.url.ends_with("/models") {
            reply(
                200,
                r#"{"object":"list","data":[{"id":"embed-only"},{"id":"chat-model"}]}"#,
            )
        } else if String::from_utf8_lossy(&r.body).contains("\"ghost\"") {
            reply(404, r#"{"error":{"message":"model not found"}}"#)
        } else {
            reply(200, CANNED)
        }
    });
    let k = kernel(
        stub.clone(),
        registry(
            r#"{ "default": "s", "providers": { "s": {
            "base_url": "http://localhost:8000/v1", "model": "ghost",
            "caps": { "vendor": "mlx" } } } }"#,
        ),
    );
    let out = root(&k, req("urn:llm:s:ask", &[("prompt", "hi")]));
    assert!(
        !stub.bodies().iter().any(|b| b.contains("embed-only")),
        "retried with the embedder: {:?}",
        stub.bodies()
    );
    let err = out.expect_err("no known chat model: the 404 surfaces");
    assert!(err.to_string().contains("404"), "{err}");
}

/// A cost tier outside the vocabulary loaded and then silently failed every
/// cost-based requirement (Hermes bug13).
#[test]
fn an_unknown_cost_tier_is_refused_at_load() {
    let err = Registry::from_json(
        r#"{ "default": "a", "providers": { "a": {
        "base_url": "http://localhost:11434/v1", "model": "m",
        "caps": { "cost": "freemium" } } } }"#,
    )
    .expect_err("an unknown cost tier must not load");
    let msg = format!("{err:?}");
    assert!(msg.contains("freemium") && msg.contains("`a`"), "{msg}");
}

/// A mistyped `caps` key (`batchat`) was ignored by serde, so the declared load
/// shape vanished at load time (Hermes bug14). A mistyped entry key (`modle`)
/// is the same silence one level up: it turned a pinned provider into a
/// discovering one.
#[test]
fn an_unknown_key_is_refused_at_load() {
    for (label, json) in [
        (
            "caps",
            r#"{ "default": "r", "providers": { "r": {
            "base_url": "http://localhost:8000/v1", "model": "q",
            "caps": { "vendor": "mlx", "batchat": 2 } } } }"#,
        ),
        (
            "entry",
            r#"{ "default": "r", "providers": { "r": {
            "base_url": "http://localhost:8000/v1", "modle": "q" } } }"#,
        ),
    ] {
        let err = Registry::from_json(json)
            .err()
            .unwrap_or_else(|| panic!("an unknown {label} key loaded"));
        let msg = format!("{err:?}");
        assert!(msg.contains("unknown field"), "{label}: {msg}");
    }
    // Both spellings of the load shape still load.
    for key in ["batchAt", "batch_at"] {
        let reg = registry(&format!(
            r#"{{ "default": "r", "providers": {{ "r": {{
            "base_url": "http://localhost:8000/v1", "model": "q", "caps": {{ "{key}": 2 }} }} }} }}"#
        ));
        assert_eq!(reg.providers[0].caps.batch_at, Some(2), "{key}");
    }
}
