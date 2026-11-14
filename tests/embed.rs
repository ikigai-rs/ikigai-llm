//! `urn:llm:embed` and `urn:llm:<provider>:embed` (ledger #3690): text to a vector
//! from a NAMED model, over Ollama's `/api/embed`, against a fake backend.
//!
//! The answer names the model's installed name and its digest, so a caller never
//! compares vectors from different models; a model that is not installed is refused
//! naming it, before the text is sent; a provider not declared local is refused
//! unless its configuration sets `embed_remote`; the answer is cacheable; and the
//! text never appears in an error.
//!
//! No real model is called: [`Fake`] answers every request in-process and records it.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Error, Expiry, Iri, Kernel, Representation, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport, Method};
use ikigai_llm::{space, OpenAiConfig, Registry};
use serde_json::{json, Value};

/// The text to embed: no answer, error or listing may carry it back.
const TEXT: &str = "INQUIRY-TEXT-41c9-never-echoed";

/// The digest the fake's `/api/tags` lists for `embeddinggemma:latest`.
const DIGEST: &str = "85462619ee721b466c5927e8a5d4fe12bd2aa1e1ea8b2c1e0b0e5a42e28b3c7e";

/// A fake Ollama: `/api/tags` lists `embeddinggemma:latest` with [`DIGEST`] (or
/// `tags`), `/api/embed` answers `embed` with `embed_status`.
struct Fake {
    tags: Value,
    embed_status: u16,
    embed: Value,
    seen: Mutex<Vec<HttpRequest>>,
}

impl Fake {
    fn new() -> Arc<Self> {
        Self::with(
            json!({ "models": [
                { "name": "llama3.2:3b", "size": 2_000_000_000u64, "digest": "aaaa" },
                { "name": "embeddinggemma:latest", "size": 600_000_000u64, "digest": DIGEST },
            ] }),
            200,
            json!({ "model": "embeddinggemma:latest", "embeddings": [[0.25, -0.5, 1.0, 0.125]] }),
        )
    }

    fn with(tags: Value, embed_status: u16, embed: Value) -> Arc<Self> {
        Arc::new(Fake {
            tags,
            embed_status,
            embed,
            seen: Mutex::new(Vec::new()),
        })
    }

    fn seen(&self, path: &str) -> Vec<HttpRequest> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.url.ends_with(path))
            .cloned()
            .collect()
    }

    fn sent_anything(&self) -> bool {
        !self.seen.lock().unwrap().is_empty()
    }
}

#[async_trait]
impl HttpTransport for Fake {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let (status, body) = match (&request.method, request.url.as_str()) {
            (Method::Get, url) if url.ends_with("/api/tags") => (200, self.tags.to_string()),
            (Method::Post, url) if url.ends_with("/api/embed") => {
                (self.embed_status, self.embed.to_string())
            }
            _ => (404, r#"{"error":"no such route"}"#.to_string()),
        };
        self.seen.lock().unwrap().push(request);
        Ok(HttpResponse {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body.into_bytes().into(),
        })
    }
}

fn kernel_over(fake: &Arc<Fake>, registry: Registry) -> Kernel {
    let transport: Arc<dyn HttpTransport> = fake.clone();
    Kernel::new(Arc::new(space(transport, registry)))
}

/// A local Ollama (`OpenAiConfig::ollama` declares `cost: local`, `vendor: ollama`).
fn kernel(fake: &Arc<Fake>) -> Kernel {
    kernel_over(fake, Registry::single(OpenAiConfig::ollama("llama3.2:3b")))
}

fn request(target: &str, args: &[(&str, &str)]) -> Request {
    let mut request = Request::new(Verb::Source, Iri::parse(target).unwrap());
    for (name, value) in args {
        request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
    }
    request
}

fn source(kernel: &Kernel, target: &str, args: &[(&str, &str)]) -> Result<Representation, Error> {
    futures::executor::block_on(kernel.issue(request(target, args), &Capability::root()))
}

fn json_of(repr: &Representation) -> Value {
    serde_json::from_slice(&repr.bytes).expect("the answer is JSON")
}

/// The error, displayed and debugged, never holding the text.
fn refusal(kernel: &Kernel, target: &str, args: &[(&str, &str)]) -> Error {
    let err = source(kernel, target, args).expect_err("must be refused");
    let text = format!("{err} / {err:?}");
    assert!(!text.contains(TEXT), "the error carries the text: {text}");
    err
}

#[test]
fn the_answer_is_the_vector_with_the_model_and_its_digest() {
    let fake = Fake::new();
    let kernel = kernel(&fake);
    let repr = source(
        &kernel,
        "urn:llm:ollama:embed",
        &[("text", TEXT), ("model", "embeddinggemma")],
    )
    .unwrap();
    assert_eq!(repr.repr_type.media_type, "application/json");
    assert_eq!(
        json_of(&repr),
        json!({
            "vector": [0.25, -0.5, 1.0, 0.125],
            // The installed name, not the caller's spelling: `embeddinggemma` and
            // `embeddinggemma:latest` are one model, and compare equal.
            "model": "embeddinggemma:latest",
            "digest": DIGEST,
            "dimensions": 4,
        })
    );
    let sent = fake.seen("/api/embed");
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url, "http://localhost:11434/api/embed");
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(
        body,
        json!({ "model": "embeddinggemma:latest", "input": TEXT })
    );
}

#[test]
fn the_facade_routes_to_the_default_provider() {
    let fake = Fake::new();
    let kernel = kernel(&fake);
    let args = [("model", "embeddinggemma:latest"), ("content", TEXT)];
    let via_facade = json_of(&source(&kernel, "urn:llm:embed", &args).unwrap());
    assert_eq!(via_facade["digest"], DIGEST);
    assert_eq!(via_facade["dimensions"], 4);
    let named = [
        ("provider", "ollama"),
        ("model", "embeddinggemma"),
        ("text", TEXT),
    ];
    assert_eq!(
        json_of(&source(&kernel, "urn:llm:embed", &named).unwrap())["vector"],
        via_facade["vector"]
    );
}

#[test]
fn a_model_that_is_not_installed_is_refused_naming_it_and_nothing_is_embedded() {
    let fake = Fake::new();
    let kernel = kernel(&fake);
    let err = refusal(
        &kernel,
        "urn:llm:ollama:embed",
        &[("text", TEXT), ("model", "nomic-embed-text")],
    );
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    let message = err.to_string();
    assert!(
        message.contains("model `nomic-embed-text` is not installed"),
        "{message}"
    );
    assert!(
        message.contains("ollama pull nomic-embed-text"),
        "{message}"
    );
    assert!(
        fake.seen("/api/embed").is_empty(),
        "the text was never sent"
    );
}

#[test]
fn the_model_is_always_named() {
    let fake = Fake::new();
    let kernel = kernel(&fake);
    for args in [vec![("text", TEXT)], vec![("text", TEXT), ("model", " ")]] {
        let err = refusal(&kernel, "urn:llm:ollama:embed", &args);
        assert!(
            matches!(&err, Error::MissingArgument(name) if name == "model"),
            "{err:?}"
        );
    }
    let err = refusal(
        &kernel,
        "urn:llm:ollama:embed",
        &[("model", "embeddinggemma")],
    );
    assert!(
        matches!(&err, Error::MissingArgument(name) if name == "text"),
        "{err:?}"
    );
    assert!(!fake.sent_anything(), "a missing input sends nothing");
}

#[test]
fn a_provider_not_declared_local_is_refused_unless_its_configuration_allows_it() {
    let registry = |extra: &str| {
        Registry::from_json(&format!(
            r#"{{ "default": "lan", "providers": {{ "lan": {{
                "base_url": "http://10.0.0.5:11434/v1",
                "caps": {{ "vendor": "ollama", "cost": "cheap" }} {extra} }} }} }}"#
        ))
        .unwrap()
    };
    let args = [("text", TEXT), ("model", "embeddinggemma")];

    let fake = Fake::new();
    let kernel = kernel_over(&fake, registry(""));
    let err = refusal(&kernel, "urn:llm:lan:embed", &args);
    assert!(matches!(err, Error::Denied(_)), "{err:?}");
    assert!(err.to_string().contains("embed_remote: true"), "{err}");
    assert!(!fake.sent_anything(), "the text never left");

    let fake = Fake::new();
    let kernel = kernel_over(&fake, registry(r#", "embed_remote": true"#));
    let answer = json_of(&source(&kernel, "urn:llm:lan:embed", &args).unwrap());
    assert_eq!(answer["digest"], DIGEST);
    assert_eq!(
        fake.seen("/api/embed")[0].url,
        "http://10.0.0.5:11434/api/embed"
    );
}

#[test]
fn the_answer_is_cached_on_the_text_and_the_model() {
    let fake = Fake::new();
    let kernel = kernel(&fake);
    let args = [("text", TEXT), ("model", "embeddinggemma")];
    for _ in 0..2 {
        let repr = source(&kernel, "urn:llm:ollama:embed", &args).unwrap();
        assert_eq!(repr.expiry, Expiry::Never);
        let threads: Vec<String> = repr.threads().iter().map(|t| t.to_string()).collect();
        assert!(threads.iter().any(|t| t == "urn:llm:config"), "{threads:?}");
        assert!(
            threads.iter().any(|t| t == "urn:llm:ollama:installed"),
            "{threads:?}"
        );
    }
    assert_eq!(
        fake.seen("/api/embed").len(),
        1,
        "the second was a cache hit"
    );
    assert!(kernel.is_cached(&request("urn:llm:ollama:embed", &args), &Capability::root()));

    // Another text, or another model, is another answer.
    source(
        &kernel,
        "urn:llm:ollama:embed",
        &[("text", "other text"), ("model", "embeddinggemma")],
    )
    .unwrap();
    assert_eq!(fake.seen("/api/embed").len(), 2);

    // A re-pull is cut by the provider's installed thread.
    kernel.cut("urn:llm:ollama:installed");
    assert!(!kernel.is_cached(&request("urn:llm:ollama:embed", &args), &Capability::root()));
}

#[test]
fn a_backend_error_keeps_its_reason_and_drops_the_text() {
    // Ollama refuses a chat model this way; a server that echoes the request
    // beside its reason contributes only the reason.
    let fake = Fake::with(
        Fake::new().tags.clone(),
        400,
        json!({ "error": format!("\"llama3.2:3b\" does not support embeddings: {TEXT}") }),
    );
    let kernel = kernel(&fake);
    let err = refusal(
        &kernel,
        "urn:llm:ollama:embed",
        &[("text", TEXT), ("model", "llama3.2:3b")],
    );
    assert!(matches!(err, Error::Endpoint(_)), "{err:?}");
    let message = err.to_string();
    assert!(message.contains("does not support embeddings"), "{message}");
    assert!(message.contains("[prompt]"), "{message}");
}

#[test]
fn an_answer_without_one_vector_or_a_listing_without_a_digest_is_refused() {
    let fake = Fake::with(Fake::new().tags.clone(), 200, json!({ "embeddings": [] }));
    let err = refusal(
        &kernel(&fake),
        "urn:llm:ollama:embed",
        &[("text", TEXT), ("model", "embeddinggemma")],
    );
    assert!(matches!(err, Error::Endpoint(_)), "{err:?}");
    assert!(err.to_string().contains("no single vector"), "{err}");

    let fake = Fake::with(
        json!({ "models": [{ "name": "embeddinggemma:latest" }] }),
        200,
        Fake::new().embed.clone(),
    );
    let err = refusal(
        &kernel(&fake),
        "urn:llm:ollama:embed",
        &[("text", TEXT), ("model", "embeddinggemma")],
    );
    assert!(err.to_string().contains("without a digest"), "{err}");
    assert!(fake.seen("/api/embed").is_empty(), "nothing was embedded");
}

#[test]
fn only_a_provider_on_ollamas_native_api_embeds() {
    let fake = Fake::new();
    let mut mlx = OpenAiConfig::discovering("mlx", "http://localhost:8000/v1");
    mlx.caps.cost = Some("local".to_string());
    mlx.caps.vendor = Some("mlx".to_string());

    // No embedding provider: neither the backend nor the facade is bound.
    let kernel = kernel_over(&fake, Registry::single(mlx.clone()));
    for target in ["urn:llm:embed", "urn:llm:mlx:embed"] {
        let err = refusal(&kernel, target, &[("text", TEXT), ("model", "m")]);
        assert!(matches!(err, Error::Unresolved(_)), "{target}: {err:?}");
    }

    // Beside one that does, naming the other is refused, listing the one that does.
    let mut registry = Registry::single(mlx);
    registry.providers.push(OpenAiConfig::ollama("llama3.2:3b"));
    let kernel = kernel_over(&fake, registry);
    let err = refusal(
        &kernel,
        "urn:llm:embed",
        &[("provider", "mlx"), ("text", TEXT), ("model", "m")],
    );
    assert!(
        matches!(&err, Error::InvalidArgument { name, .. } if name == "provider"),
        "{err:?}"
    );
    assert!(
        err.to_string().contains("providers that do: ollama"),
        "{err}"
    );
    // The default (`mlx`) does not embed either: refused, saying how to choose.
    let err = refusal(&kernel, "urn:llm:embed", &[("text", TEXT), ("model", "m")]);
    assert!(err.to_string().contains("provider="), "{err}");
    assert!(!fake.sent_anything());
}

#[test]
fn a_caller_that_cannot_reach_the_host_is_denied_before_anything_is_sent() {
    let fake = Fake::new();
    let kernel = kernel(&fake);
    let other = Capability::scoped(["urn:cap:net:example.com".to_string()]);
    let err = futures::executor::block_on(kernel.issue(
        request(
            "urn:llm:ollama:embed",
            &[("text", TEXT), ("model", "embeddinggemma")],
        ),
        &other,
    ))
    .unwrap_err();
    assert!(matches!(err, Error::Denied(_)), "{err:?}");
    assert!(err.to_string().contains("localhost"), "{err}");
    assert!(!fake.sent_anything());
}
