//! A provider's key kept in ikigai-secret, not inline in the registry (ledger
//! #2691). `api_key_secret` names a secret; the backend reads it at request time
//! through the host's `SecretReader`, for a caller holding the provider's use
//! scope `urn:cap:llm:use:<provider>` — never under the caller's own authority,
//! so the grant that lets a caller use the key never lets it read the key
//! (ledger #2706).
//!
//! The secrets space is the real one (`ikigai_secret::space`) over its file store
//! in a scratch directory, mounted beside llm so a test can try to read the key
//! the way a caller would; the host's reader is [`HostSecrets`] over the same
//! store; the provider is [`Stub`], an in-process
//! [`HttpTransport`] that answers like an OpenAI-shaped API (401 without the right
//! Bearer token) and logs every request, so a test can prove what was sent.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ikigai_core::{
    ArgRef, Capability, EndpointSpace, Error, Fallback, Iri, Kernel, Representation, Request,
    Space, Verb,
};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, space_with_secrets, use_scope, OpenAiConfig, Registry, SecretReader};
use ikigai_secret::{Backend, FileBackend};

/// The secret's value: a shape every provider-issued key has.
const KEY: &str = "sk-proj-Secr3tValue_0123456789";
const SECRET: &str = "remote-api-key";

const CANNED: &str = r#"{"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"Hello there!"},"finish_reason":"stop"}]}"#;

struct Stub {
    log: Mutex<Vec<HttpRequest>>,
}

impl Stub {
    fn new() -> Arc<Self> {
        Arc::new(Stub {
            log: Mutex::new(Vec::new()),
        })
    }

    fn sent(&self) -> Vec<HttpRequest> {
        self.log.lock().unwrap().clone()
    }
}

fn reply(status: u16, body: &str) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![],
        body: body.as_bytes().to_vec().into(),
    }
}

fn authorization(r: &HttpRequest) -> Vec<String> {
    r.headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| v.clone())
        .collect()
}

#[async_trait]
impl HttpTransport for Stub {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let authed = authorization(&request) == [format!("Bearer {KEY}")];
        let out = if !authed {
            reply(
                401,
                r#"{"error":{"message":"You didn't provide an API key."}}"#,
            )
        } else if request.url.ends_with("/models") {
            reply(200, r#"{"object":"list","data":[{"id":"gpt-4o"}]}"#)
        } else {
            reply(200, CANNED)
        };
        self.log.lock().unwrap().push(request);
        Ok(out)
    }
}

/// The host's reader: its own keystore, read on its own authority, as a host
/// bridges `ikigai_secret::Backend` (ikigai-embedded's `HostSecrets` for
/// ikigai-meeting is the same few lines).
struct HostSecrets(Arc<FileBackend>);

impl SecretReader for HostSecrets {
    fn read(&self, name: &str) -> ikigai_core::Result<Vec<u8>> {
        self.0
            .get(name)?
            .ok_or_else(|| Error::NotFound(format!("no secret `{name}`")))
    }
}

/// A scratch secrets store, removed when dropped.
struct Store(PathBuf);

impl Store {
    fn new(test: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ikigai-llm-secret-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Store(dir)
    }

    fn backend(&self) -> Arc<FileBackend> {
        Arc::new(FileBackend::new(self.0.clone()))
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const NAMED: &str = r#"{ "default": "remote", "providers": {
    "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o",
                "api_key_secret": "remote-api-key" } } }"#;

const NAMED_DISCOVERING: &str = r#"{ "default": "remote", "providers": {
    "remote": { "base_url": "https://api.example.com/v1",
                "api_key_secret": "remote-api-key" } } }"#;

/// The llm space, reading keys through the host's reader, with the real secrets
/// space beside it, in one kernel.
fn kernel(stub: Arc<Stub>, store: &Store, json: &str) -> Kernel {
    let transport: Arc<dyn HttpTransport> = stub;
    let llm: EndpointSpace = space_with_secrets(
        transport,
        Arc::new(HostSecrets(store.backend())),
        Registry::from_json(json).unwrap(),
    );
    let secrets = ikigai_secret::space(store.backend());
    let spaces: Vec<Arc<dyn Space>> = vec![Arc::new(llm), Arc::new(secrets)];
    Kernel::new(Arc::new(Fallback::new(spaces)))
}

fn ask(iri: &str) -> Request {
    Request::new(Verb::Source, Iri::parse(iri).unwrap())
        .with_arg("prompt", ArgRef::Inline(b"hi".to_vec()))
}

fn get(iri: &str) -> Request {
    Request::new(Verb::Source, Iri::parse(iri).unwrap())
}

fn run(k: &Kernel, r: Request, cap: &Capability) -> Result<Representation, Error> {
    futures::executor::block_on(k.issue(r, cap))
}

fn text(rep: &Representation) -> String {
    String::from_utf8_lossy(&rep.bytes).into_owned()
}

fn scoped(scopes: &[&str]) -> Capability {
    Capability::scoped(scopes.iter().map(|s| s.to_string()))
}

/// What a caller of a secret-keyed provider holds: the host, and the use of the
/// provider. No `urn:cap:secret:*` grant at all.
fn granted() -> Capability {
    scoped(&["urn:cap:net:api.example.com", "urn:cap:llm:use:remote"])
}

#[test]
fn a_provider_naming_a_secret_authenticates_with_it() {
    let store = Store::new("auth");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED);

    for iri in ["urn:llm:remote:ask", "urn:llm:ask"] {
        let rep = run(&k, ask(iri), &granted()).unwrap_or_else(|e| panic!("{iri}: {e}"));
        assert_eq!(text(&rep), "Hello there!", "{iri}");
    }
    let sent = stub.sent();
    assert_eq!(sent.len(), 2);
    for r in &sent {
        assert_eq!(authorization(r), [format!("Bearer {KEY}")], "{}", r.url);
    }
}

/// A secret written with `echo` ends in a newline; the key sent does not.
#[test]
fn a_trailing_newline_in_the_secret_is_not_sent() {
    let store = Store::new("newline");
    store
        .backend()
        .set(SECRET, format!("{KEY}\n").as_bytes())
        .unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED);
    let rep = run(&k, ask("urn:llm:remote:ask"), &granted()).unwrap();
    assert_eq!(text(&rep), "Hello there!");
}

/// The listing, liveness and discovery probes carry the key too: an
/// OpenAI-shaped `/models` answers 401 without it.
#[test]
fn the_probes_carry_the_secret_key() {
    let store = Store::new("probes");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED_DISCOVERING);

    assert_eq!(
        text(&run(&k, get("urn:llm:remote:up"), &granted()).unwrap()),
        "true"
    );
    assert_eq!(
        text(&run(&k, get("urn:llm:remote:installed"), &granted()).unwrap()),
        "gpt-4o"
    );
    assert_eq!(
        text(&run(&k, get("urn:llm:remote:model"), &granted()).unwrap()),
        "gpt-4o"
    );
    assert_eq!(
        text(&run(&k, ask("urn:llm:remote:ask"), &granted()).unwrap()),
        "Hello there!"
    );
    assert!(!stub.sent().is_empty());
    for r in stub.sent() {
        assert_eq!(authorization(&r), [format!("Bearer {KEY}")], "{}", r.url);
    }
}

#[test]
fn a_missing_secret_refuses_naming_the_secret_and_sends_nothing() {
    let store = Store::new("missing");
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED);

    let err = run(&k, ask("urn:llm:remote:ask"), &granted()).unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    let message = err.to_string();
    assert!(message.contains(SECRET), "names the secret: {message}");
    assert!(message.contains("remote"), "names the provider: {message}");
    assert!(stub.sent().is_empty(), "nothing went out without the key");
}

/// A caller who may reach the host but not use the provider is refused before
/// the key is read, and the error carries no part of the stored value. Holding
/// the secret's read grant instead does not stand in for the use scope.
#[test]
fn a_caller_without_the_use_scope_is_denied_and_never_sees_the_key() {
    let store = Store::new("denied");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED);

    let err = run(
        &k,
        ask("urn:llm:remote:ask"),
        &scoped(&["urn:cap:net:api.example.com"]),
    )
    .unwrap_err();
    assert!(matches!(err, Error::Denied(_)), "{err:?}");
    let message = err.to_string();
    assert!(
        message.contains("urn:cap:llm:use:remote"),
        "names the scope: {message}"
    );
    assert!(!message.contains(KEY), "the key leaked: {message}");
    assert!(stub.sent().is_empty());

    // Neither the use of a DIFFERENT provider nor the old read grant on the
    // secret stands in for this provider's use scope; the facade routes to the
    // same backend and is refused the same way.
    for (iri, held) in [
        ("urn:llm:remote:ask", "urn:cap:llm:use:other"),
        ("urn:llm:remote:ask", "urn:cap:secret:read:remote-api-key"),
        ("urn:llm:ask", "urn:cap:secret:read:remote-api-key"),
    ] {
        let err = run(
            &k,
            ask(iri),
            &scoped(&["urn:cap:net:api.example.com", held]),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Denied(_)), "{iri} {held}: {err:?}");
        assert!(!err.to_string().contains(KEY), "the key leaked: {err}");
    }
    assert!(stub.sent().is_empty());
}

/// Declared = enforced: every action that sends a keyed provider's key declares
/// its use scope beside the net floor, and none declares a secret grant. An
/// unkeyed provider declares no use scope.
#[test]
fn a_keyed_providers_actions_declare_its_use_scope_and_no_secret_grant() {
    let store = Store::new("declared");
    let json = r#"{ "default": "remote", "providers": {
        "remote": { "base_url": "https://api.example.com/v1",
                    "api_key_secret": "remote-api-key" },
        "local":  { "base_url": "http://localhost:11434/v1" } } }"#;
    let k = kernel(Stub::new(), &store, json);
    assert_eq!(use_scope("remote"), "urn:cap:llm:use:remote");
    let requires = |iri: String| -> Vec<String> {
        k.describe(&Iri::parse(&iri).unwrap())
            .unwrap()
            .action_specs()
            .into_iter()
            .flat_map(|s| s.requires)
            .collect()
    };
    for face in ["ask", "up", "installed", "model"] {
        let keyed = requires(format!("urn:llm:remote:{face}"));
        assert_eq!(
            keyed,
            ["urn:cap:net:*", "urn:cap:llm:use:remote"],
            "{face}: declared = enforced"
        );
        let unkeyed = requires(format!("urn:llm:local:{face}"));
        assert_eq!(unkeyed, ["urn:cap:net:*"], "local {face}");
    }
}

/// The inventory probes every provider without declaring each one's scope: a
/// caller without the use scope is not refused the inventory, the keyed
/// provider is simply not probed (nothing is sent, no key is read).
#[test]
fn the_inventory_does_not_probe_a_keyed_provider_the_caller_cannot_use() {
    let store = Store::new("inventory");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED_DISCOVERING);
    let body = text(
        &run(
            &k,
            get("urn:llm:models"),
            &scoped(&["urn:cap:net:api.example.com"]),
        )
        .unwrap(),
    );
    assert!(body.contains("remote"), "{body}");
    assert!(!body.contains(KEY));
    assert!(stub.sent().is_empty(), "probed without the use scope");
}

/// A keyed provider mounted with `space` (no reader) refuses, naming the
/// provider and the secret: it never falls back to the caller's authority.
#[test]
fn a_keyed_provider_without_a_reader_refuses_and_sends_nothing() {
    let store = Store::new("noreader");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let stub = Stub::new();
    let transport: Arc<dyn HttpTransport> = stub.clone();
    let spaces: Vec<Arc<dyn Space>> = vec![
        Arc::new(space(transport, Registry::from_json(NAMED).unwrap())),
        Arc::new(ikigai_secret::space(store.backend())),
    ];
    let k = Kernel::new(Arc::new(Fallback::new(spaces)));
    let caller = scoped(&[
        "urn:cap:net:api.example.com",
        "urn:cap:llm:use:remote",
        "urn:cap:secret:read:remote-api-key",
    ]);
    let err = run(&k, ask("urn:llm:remote:ask"), &caller).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("SecretReader"), "{message}");
    assert!(
        message.contains(SECRET) && message.contains("remote"),
        "{message}"
    );
    assert!(stub.sent().is_empty());
}

/// A secret that is not a Bearer token is refused before anything is sent,
/// without quoting it.
#[test]
fn a_secret_that_is_not_a_token_is_refused_without_echoing_it() {
    let store = Store::new("malformed");
    let bad = "sk-bad\r\nX-Injected: 1";
    store.backend().set(SECRET, bad.as_bytes()).unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED);

    let err = run(&k, ask("urn:llm:remote:ask"), &granted()).unwrap_err();
    let message = err.to_string();
    assert!(!message.contains("sk-bad"), "the key leaked: {message}");
    assert!(!message.contains("Injected"), "the key leaked: {message}");
    assert!(stub.sent().is_empty());
}

/// `urn:llm:config` reports the secret's NAME (not a secret) and no value.
#[test]
fn the_config_face_names_the_secret_and_holds_no_key() {
    let store = Store::new("config");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let k = kernel(Stub::new(), &store, NAMED);
    let body = text(&run(&k, get("urn:llm:config"), &Capability::root()).unwrap());
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["providers"]["remote"]["api_key_secret"], SECRET);
    assert_eq!(v["providers"]["remote"]["api_key"], serde_json::Value::Null);
    assert!(!body.contains(KEY));
}

#[test]
fn a_registry_naming_a_key_twice_or_badly_is_refused() {
    let both = r#"{ "default": "r", "providers": { "r": {
        "base_url": "https://api.example.com/v1", "api_key": "sk-inline",
        "api_key_secret": "r-key" } } }"#;
    let err = Registry::from_json(both).unwrap_err().to_string();
    assert!(
        err.contains("`r`") && err.contains("api_key_secret"),
        "{err}"
    );
    assert!(!err.contains("sk-inline"), "the key leaked: {err}");

    for bad in ["", "-r-key", "r key", "r/key", "r?key", "r#key"] {
        let json = format!(
            r#"{{ "default": "r", "providers": {{ "r": {{
            "base_url": "https://api.example.com/v1", "api_key_secret": "{bad}" }} }} }}"#
        );
        let err = Registry::from_json(&json).unwrap_err().to_string();
        assert!(err.contains("cannot name a secret"), "{bad:?}: {err}");
    }
}

/// The inline form keeps working (deprecated, logged at load).
#[test]
fn an_inline_key_still_loads_and_authenticates() {
    let json = format!(
        r#"{{ "default": "remote", "providers": {{ "remote": {{
        "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "{KEY}" }} }} }}"#
    );
    let registry = Registry::from_json(&json).unwrap();
    assert_eq!(registry.providers[0].api_key.as_deref(), Some(KEY));
    assert_eq!(registry.providers[0].api_key_secret, None);
    let stub = Stub::new();
    let transport: Arc<dyn HttpTransport> = stub.clone();
    let k = Kernel::new(Arc::new(space(transport, registry)));
    let rep = run(
        &k,
        ask("urn:llm:remote:ask"),
        &scoped(&["urn:cap:net:api.example.com"]),
    )
    .unwrap();
    assert_eq!(text(&rep), "Hello there!");
}

/// A provider built in code may name a secret too.
#[test]
fn a_provider_built_in_code_may_name_a_secret() {
    let store = Store::new("code");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let config = OpenAiConfig {
        base_url: "https://api.example.com/v1".to_string(),
        api_key_secret: Some(SECRET.to_string()),
        secrets: Some(Arc::new(HostSecrets(store.backend()))),
        ..OpenAiConfig::discovering("remote", "unused")
    };
    let stub = Stub::new();
    let transport: Arc<dyn HttpTransport> = stub.clone();
    let llm = space(transport, Registry::single(config));
    let spaces: Vec<Arc<dyn Space>> = vec![
        Arc::new(llm),
        Arc::new(ikigai_secret::space(store.backend())),
    ];
    let k = Kernel::new(Arc::new(Fallback::new(spaces)));
    let rep = run(&k, ask("urn:llm:remote:ask"), &granted()).unwrap();
    assert_eq!(text(&rep), "Hello there!");
}

/// Ledger #2706: the grants a model caller needs must not hand it the key. A
/// caller holding exactly what this crate tells it to hold (and no
/// `urn:cap:secret:read`) asks the provider, then asks ikigai-secret for the
/// same secret directly, and is refused. Before the fix the README told it to
/// hold `urn:cap:secret:read:<name>`, and this read returned the key.
#[test]
fn the_grants_a_model_caller_needs_cannot_read_the_key() {
    let store = Store::new("repro");
    store.backend().set(SECRET, KEY.as_bytes()).unwrap();
    let stub = Stub::new();
    let k = kernel(Arc::clone(&stub), &store, NAMED);

    let rep = run(&k, ask("urn:llm:remote:ask"), &granted()).unwrap();
    assert_eq!(text(&rep), "Hello there!");

    match run(&k, get("urn:secret:remote-api-key"), &granted()) {
        Ok(rep) => panic!("the caller read the raw key: {:?}", text(&rep)),
        Err(e) => assert!(matches!(e, Error::Denied(_)), "{e:?}"),
    }
}
