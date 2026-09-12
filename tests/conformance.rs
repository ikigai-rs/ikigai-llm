//! The module recipe as one test: `ikigai-conformance` walks every endpoint
//! [`ikigai_llm::space`] binds and reports every violation at once.
//!
//! ## The fixture kernel: a loopback stub speaking Ollama's API
//!
//! Every backend action reaches an inference server, and CI has none. So
//! [`Stub`] is an HTTP/1.1 listener on `127.0.0.1` at an ephemeral port that
//! answers the four routes this module speaks — the OpenAI-compatible
//! `POST /v1/chat/completions` and `GET /v1/models`, and Ollama's native
//! `GET /api/tags` and `POST /api/show` — with a canned model, `tiny:1b`, that
//! never echoes a prompt. It records every request and counts every
//! connection it accepts: the count is how a test proves a socket was NEVER
//! opened. [`Client`] is the smallest [`HttpTransport`] that speaks to it (one
//! blocking exchange per request), injected the way a host injects `ureq`.
//! Every provider's `base_url` points at the stub, so the fired actions run
//! under root against it.
//!
//! ## Two registries, two walks
//!
//! The registry decides what is cacheable, so the suite walks two of them:
//!
//! - [`conforms`] walks a **pinned** registry (one `ollama` provider naming its
//!   model) and declares `llm-config`, `llm-models`, `llm-select` and
//!   `llm-ollama-model` `cacheable`: each is a function of the registry, held
//!   to a cache hit on the second resolution and to a non-empty thread set —
//!   `urn:llm:config`, the registry's golden thread, which
//!   [`config_derived_results_are_cut_by_the_registry_thread`] cuts by hand.
//! - [`conforms_with_a_discovering_provider`] adds a provider that names only
//!   its **server**. Its `:model` probes the backend (so it declares the net
//!   capability and is live), and one discovering provider makes the inventory
//!   and selection live facts too — so nothing is declared cacheable there. The
//!   declaration is true over the REGISTRY walked, not the module in isolation
//!   (conformance PENDING #18/#30/#47).
//!
//! Nothing is `pure`: every cacheable result reads the registry. No opt-outs:
//! every action fires against the stub. No module namespace: the Turtle face
//! uses `ik:` terms the shared vocabulary defines — every one of them, load
//! shape included ([`a_declared_load_shape_introduces_no_undefined_term`]).
//! NAMES runs: every id is kebab-case.
//!
//! ## What the suite cannot see, pinned by hand
//!
//! - **ENFORCED never reaches a socket**
//!   ([`every_network_action_is_denied_before_any_socket_opens`]): the suite
//!   sees the typed `Denied`; the stub's connection count sees that nothing
//!   connected. Two capability shapes the suite cannot form (PENDING #46): no
//!   grants (the KERNEL's floor refuses on the declared `urn:cap:net:*`) and a
//!   grant on ANOTHER host (the wildcard admits the call and the module's own
//!   per-host rule refuses — typed `Denied` too, for every action, the facade
//!   and a discovering `:model` included).
//! - **An answer is never cacheable** ([`answers_are_live_by_construction`],
//!   PENDING #22/#64): generation is non-deterministic, so `:ask` is live;
//!   liveness and a discovered identity are live facts. The suite's cache probe
//!   returns early on `Expiry::Always` and cannot tell a decision from an
//!   omission; the stub's hit count can — two resolutions, two requests.
//! - **The registry thread cuts every config-derived result**
//!   ([`config_derived_results_are_cut_by_the_registry_thread`]): CACHEABLE
//!   checks the thread set is non-empty, not that a cut works.
//! - **Declared outputs against what is served** (PENDING #11/#31,
//!   [`declared_outputs_are_the_media_types_served`]): every `as=` face served
//!   is declared, every declared face is served by some call.
//! - **A prompt never appears in the errors this module composes** (PENDING
//!   #67, [`a_prompt_never_appears_in_the_errors_this_module_composes`]): the
//!   denial, a bad `needs=`, a no-match, a discovery failure, and a backend
//!   4xx — where a server that echoes the request beside its `error.message`
//!   still contributes only the message.
//! - **The manifold states the contract**
//!   ([`the_manifold_states_the_contract`]): every input has a class, every
//!   action that reaches the network declares exactly `urn:cap:net:*` and the
//!   config reads declare nothing, only `prompt` / `needs` are required (no
//!   check can see "required but actually optional", PENDING #5/#49).

use async_trait::async_trait;
use ikigai_conformance::{Checks, Fixture, Report, Suite};
use ikigai_core::{ArgRef, Capability, Error, Expiry, Iri, Kernel, Representation, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{OpenAiConfig, Registry};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use url::Url;

/// The model the stub serves, everywhere it is asked.
const MODEL: &str = "tiny:1b";

/// The scope that admits the stub, and one that admits a host it is not.
const STUB_SCOPE: &str = "urn:cap:net:127.0.0.1";
const OTHER_SCOPE: &str = "urn:cap:net:example.com";

/// The wildcard every network action must declare: "holds some grant under
/// this prefix".
const NET_WILDCARD: &str = "urn:cap:net:*";

/// The registry's golden thread.
const CONFIG_THREAD: &str = "urn:llm:config";

/// A prompt no stub answer, no description and no error may contain.
const PROMPT: &str = "SECRET-PROMPT-7f3a-never-echoed";

/// One request as the stub received it.
#[derive(Clone, Debug)]
struct Received {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// One HTTP/1.x message off a stream: the start line, the headers, and a body
/// of `Content-Length` bytes (or to EOF).
struct Message {
    start: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn read_message(stream: &mut TcpStream) -> Option<Message> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.lines();
    let start = lines.next()?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end..].to_vec();
    while body.len() < len {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(len);
    Some(Message {
        start,
        headers,
        body,
    })
}

/// A loopback inference server on an ephemeral port speaking Ollama's API
/// shape, recording what it receives and counting what it accepts.
struct Stub {
    addr: SocketAddr,
    connections: Arc<AtomicUsize>,
    received: Arc<Mutex<Vec<Received>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Stub {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
        let addr = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let received = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (connections, received, stop) =
                (connections.clone(), received.clone(), stop.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    connections.fetch_add(1, Ordering::SeqCst);
                    let Some(Message { start, body, .. }) = read_message(&mut stream) else {
                        continue;
                    };
                    let mut parts = start.split_whitespace();
                    let method = parts.next().unwrap_or("").to_string();
                    let path = parts.next().unwrap_or("/").to_string();
                    let (status, reason, body_out) = respond(&method, &path, &body);
                    received
                        .lock()
                        .unwrap()
                        .push(Received { method, path, body });
                    let head = format!(
                        "HTTP/1.1 {status} {reason}\r\nConnection: close\r\n\
                         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                        body_out.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&body_out);
                    let _ = stream.flush();
                }
            })
        };
        Stub {
            addr,
            connections,
            received,
            stop,
            thread: Some(thread),
        }
    }

    /// A `base_url` on this stub, by the address the capability is granted on.
    fn base(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.addr.port())
    }

    fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    fn received(&self) -> Vec<Received> {
        self.received.lock().unwrap().clone()
    }

    /// How many requests reached `path`.
    fn hits(&self, path: &str) -> usize {
        self.received().iter().filter(|r| r.path == path).count()
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock the accept loop so the thread sees the flag.
        let _ = TcpStream::connect(self.addr);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The stub's script: Ollama's four routes under `/v1` (the OpenAI-compatible
/// face) and `/api` (the native face); under `/echo/v1` a 400 whose body
/// carries the request back beside its `error.message`; anything else a 404.
fn respond(method: &str, path: &str, request: &[u8]) -> (u16, &'static str, Vec<u8>) {
    let ok = |body: &str| (200, "OK", body.as_bytes().to_vec());
    match (method, path) {
        ("POST", "/v1/chat/completions") => ok(&format!(
            r#"{{"model":"{MODEL}","choices":[{{"message":{{"role":"assistant","content":"Hello there!"}},"finish_reason":"stop"}}],"usage":{{"total_tokens":7}}}}"#
        )),
        ("GET", "/v1/models") => ok(&format!(r#"{{"object":"list","data":[{{"id":"{MODEL}"}}]}}"#)),
        ("GET", "/api/tags") => ok(&format!(
            r#"{{"models":[{{"name":"{MODEL}","size":1000}}]}}"#
        )),
        ("POST", "/api/show") => ok(
            r#"{"details":{"parameter_size":"1B"},"model_info":{"llama.context_length":4096},"capabilities":["completion"]}"#,
        ),
        ("POST", "/echo/v1/chat/completions") => (
            400,
            "Bad Request",
            format!(
                r#"{{"error":{{"message":"bad request","type":"invalid_request_error"}},"echo":{}}}"#,
                serde_json::Value::String(String::from_utf8_lossy(request).into_owned())
            )
            .into_bytes(),
        ),
        _ => (
            404,
            "Not Found",
            br#"{"error":{"message":"no such path","type":"not_found"}}"#.to_vec(),
        ),
    }
}

/// The smallest transport that speaks to the stub: one blocking HTTP/1.1
/// exchange per request.
struct Client;

#[async_trait]
impl HttpTransport for Client {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let url = Url::parse(&request.url).map_err(|e| e.to_string())?;
        let host = url.host_str().ok_or("no host")?;
        let port = url.port_or_known_default().ok_or("no port")?;
        let mut target = url.path().to_string();
        if let Some(q) = url.query() {
            target.push('?');
            target.push_str(q);
        }
        let mut stream = TcpStream::connect((host, port)).map_err(|e| e.to_string())?;
        let mut head = format!(
            "{} {target} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nContent-Length: {}\r\n",
            request.method.as_str(),
            request.body.len()
        );
        for (k, v) in &request.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        head.push_str("\r\n");
        stream
            .write_all(head.as_bytes())
            .and_then(|()| stream.write_all(&request.body))
            .map_err(|e| e.to_string())?;
        let Message {
            start,
            headers,
            body,
        } = read_message(&mut stream).ok_or("no response")?;
        let status = start
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad status line `{start}`"))?;
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

/// A pinned `ollama` provider on the stub: names its model, declares the
/// traits selection reads (`cost=local` is the fixture's `needs=`), and its
/// `vendor: ollama` opts into native `/api/tags` + `/api/show` discovery.
fn ollama(stub: &Stub, base: &str) -> OpenAiConfig {
    let mut config = OpenAiConfig::ollama(MODEL);
    config.base_url = stub.base(base);
    config.caps.context = Some(4096);
    config.caps.modalities = vec!["text".to_string()];
    config
}

fn pinned(stub: &Stub) -> Registry {
    Registry::single(ollama(stub, "/v1"))
}

/// The pinned registry plus a `server` provider that names only its server:
/// the model is discovered from `GET /v1/models` per resolve.
fn discovering(stub: &Stub) -> Registry {
    let mut registry = pinned(stub);
    let mut server = OpenAiConfig::discovering("server", stub.base("/v1"));
    server.caps.cost = Some("local".to_string());
    server.caps.vendor = Some("mlx".to_string());
    registry.providers.push(server);
    registry
}

fn kernel(registry: Registry) -> Kernel {
    Kernel::new(Arc::new(ikigai_llm::space(Arc::new(Client), registry)))
}

/// The suite for this module: `llm-select` needs a `needs=` the grammar
/// admits (the suite's `x` is a grammar error, PENDING #7/#33); everything
/// else resolves with the minimal inputs its ArgSpecs allow.
fn suite() -> Suite {
    Suite::new().fixture(Fixture::new("llm-select", Verb::Source).arg("needs", "cost=local"))
}

/// The four endpoints every registry binds, then four per provider.
fn endpoints(providers: usize) -> usize {
    4 + 4 * providers
}

/// The walk saw every endpoint, one Source action each, skipped no check, and
/// opted nothing out. An endpoint bound without a line here is held to a
/// weaker standard.
fn assert_shape(report: &Report, providers: usize) {
    assert_eq!(report.endpoints, endpoints(providers), "{report}");
    assert_eq!(
        report.actions,
        endpoints(providers),
        "one Source action each: {report}"
    );
    assert_eq!(
        report.checks.skipped().count(),
        0,
        "every check runs: {report}"
    );
    assert!(report.declared.opted_out.is_empty(), "{report}");
    assert!(report.declared.pure.is_empty(), "{report}");
}

/// The stub saw only the routes this module speaks, and each `ask` reached it
/// exactly once per action — the facade's and the backend's — because an
/// answer is never served from cache.
fn assert_stub_footprint(stub: &Stub, asks: usize) {
    for r in stub.received() {
        assert!(
            matches!(
                (r.method.as_str(), r.path.as_str()),
                ("POST", "/v1/chat/completions")
                    | ("GET", "/v1/models")
                    | ("GET", "/api/tags")
                    | ("POST", "/api/show")
            ),
            "an unscripted route was requested: {} {}",
            r.method,
            r.path
        );
    }
    assert_eq!(
        stub.hits("/v1/chat/completions"),
        asks,
        "one ask per action"
    );
}

#[test]
fn conforms() {
    let stub = Stub::start();
    let report = suite()
        .cacheable("llm-config")
        .cacheable("llm-models")
        .cacheable("llm-select")
        .cacheable("llm-ollama-model")
        .run_blocking(&kernel(pinned(&stub)));
    // Printed even when clean (`--nocapture`): the report is the record.
    eprintln!("{report}");
    assert!(report.is_clean(), "{report}");
    assert_shape(&report, 1);
    assert_eq!(
        report.declared.cacheable,
        ["llm-config", "llm-models", "llm-select", "llm-ollama-model"],
        "{report}"
    );
    assert_stub_footprint(&stub, 2);
}

#[test]
fn conforms_with_a_discovering_provider() {
    let stub = Stub::start();
    let report = suite().run_blocking(&kernel(discovering(&stub)));
    eprintln!("{report}");
    assert!(report.is_clean(), "{report}");
    assert_shape(&report, 2);
    assert!(report.declared.cacheable.is_empty(), "{report}");
    assert_stub_footprint(&stub, 3);
}

/// `ik:batchAt` is emitted only when a provider declares a load shape, so
/// neither [`conforms`] nor [`conforms_with_a_discovering_provider`] walks a
/// graph that carries it — the term would be covered by nothing. This walks
/// one that does. It is the inverse of the pin it replaces: the vocabulary
/// lacked `ik:batchAt` until 0.1.69, this asserted the finding, and defining
/// the term flipped it (core #107, vocab 0.1.69).
///
/// It is not a string check: the check PARSES the face (oxrdfio) before reading
/// its terms, so an unparseable graph is a finding too — which is what covers
/// the `@prefix xsd:` line the tagged `ik:batchAt` literal now needs.
#[test]
fn a_declared_load_shape_introduces_no_undefined_term() {
    let stub = Stub::start();
    let mut registry = pinned(&stub);
    registry.providers[0].caps.batch_at = Some(2);
    let report = suite()
        .checks(Checks::VOCABULARY)
        .run_blocking(&kernel(registry));
    eprintln!("{report}");
    assert!(report.is_clean(), "{report}");
}

fn request(verb: Verb, iri: &str, args: &[(&str, &str)]) -> Request {
    let mut request = Request::new(verb, Iri::parse(iri).unwrap());
    for (name, value) in args {
        request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
    }
    request
}

fn issue(
    kernel: &Kernel,
    iri: &str,
    args: &[(&str, &str)],
    capability: &Capability,
) -> Result<Representation, Error> {
    futures::executor::block_on(kernel.issue(request(Verb::Source, iri, args), capability))
}

fn scoped(scopes: &[&str]) -> Capability {
    Capability::scoped(scopes.iter().map(|s| s.to_string()))
}

/// Every action that reaches the network, over the discovering registry: the
/// facade, and per provider `:ask` / `:up` / `:installed`, plus the
/// discovering provider's `:model`.
const NETWORK_ACTIONS: [&str; 8] = [
    "urn:llm:ask",
    "urn:llm:ollama:ask",
    "urn:llm:ollama:up",
    "urn:llm:ollama:installed",
    "urn:llm:server:ask",
    "urn:llm:server:up",
    "urn:llm:server:installed",
    "urn:llm:server:model",
];

/// The config reads: no network, no capability.
const CONFIG_READS: [&str; 4] = [
    "urn:llm:config",
    "urn:llm:models",
    "urn:llm:select",
    "urn:llm:ollama:model",
];

/// Under no grants — and under a grant on another host — every network action
/// refuses with a typed, permanent `Denied`, and the stub accepts no
/// connection. Two gates are in play: with no net grant at all the KERNEL
/// refuses on the declared `urn:cap:net:*` before the endpoint runs; with a
/// grant on another host the wildcard admits the call and the module's
/// per-host rule refuses it, naming the host. Under a grant on the stub's host
/// the call connects.
#[test]
fn every_network_action_is_denied_before_any_socket_opens() {
    let stub = Stub::start();
    let kernel = kernel(discovering(&stub));
    for (capability, gate) in [
        (scoped(&[]), NET_WILDCARD),
        (scoped(&[OTHER_SCOPE]), "127.0.0.1"),
    ] {
        for iri in NETWORK_ACTIONS {
            let err = issue(&kernel, iri, &[("prompt", PROMPT)], &capability)
                .err()
                .unwrap_or_else(|| panic!("{iri} resolved under {capability:?}"));
            assert!(matches!(err, Error::Denied(_)), "{iri}: {err:?}");
            assert!(!err.is_transient(), "{iri}: {err:?}");
            assert!(
                err.to_string().contains(gate),
                "{iri}: the denial names what refused it: {err}"
            );
        }
        // The config reads are not gated: they answer under no grants at all.
        for iri in CONFIG_READS {
            issue(&kernel, iri, &[("needs", "cost=local")], &capability)
                .unwrap_or_else(|e| panic!("{iri}: {e}"));
        }
    }
    assert_eq!(
        stub.connections(),
        0,
        "the gate precedes the socket: nothing connected"
    );
    let up = issue(&kernel, "urn:llm:ollama:up", &[], &scoped(&[STUB_SCOPE])).unwrap();
    assert_eq!(up.bytes, b"true");
    assert_eq!(stub.connections(), 1, "a granted host connects");
}

/// An answer is never cacheable: generation is non-deterministic, so two
/// identical asks are two requests. Liveness and a discovered identity are
/// live facts. The suite cannot see this (its cache probe returns early on
/// `Expiry::Always`); the stub's hit count can.
#[test]
fn answers_are_live_by_construction() {
    let stub = Stub::start();
    let kernel = kernel(discovering(&stub));
    let root = Capability::root();
    for (iri, path, args) in [
        (
            "urn:llm:ollama:ask",
            "/v1/chat/completions",
            vec![("prompt", PROMPT), ("temperature", "0")],
        ),
        ("urn:llm:ollama:up", "/v1/models", vec![]),
        ("urn:llm:server:model", "/v1/models", vec![]),
    ] {
        let before = stub.hits(path);
        for _ in 0..2 {
            let repr = issue(&kernel, iri, &args, &root).unwrap_or_else(|e| panic!("{iri}: {e}"));
            assert_eq!(repr.expiry, Expiry::Always, "{iri} is live");
            assert!(
                repr.threads().is_empty(),
                "{iri}: a live fact names no thread"
            );
        }
        assert_eq!(
            stub.hits(path),
            before + 2,
            "{iri}: two resolutions, two requests — even at temperature 0"
        );
        assert!(!kernel.is_cached(&request(Verb::Source, iri, &args), &root));
    }
}

/// Every config-derived result is cacheable under the registry's one thread,
/// `urn:llm:config`, and cutting it evicts all of them — the hook a live
/// reload will use, and the thread a copy cached across a mount can be cut by.
#[test]
fn config_derived_results_are_cut_by_the_registry_thread() {
    let stub = Stub::start();
    let kernel = kernel(pinned(&stub));
    let root = Capability::root();
    let requests: Vec<Request> = CONFIG_READS
        .iter()
        .map(|iri| request(Verb::Source, iri, &[("needs", "cost=local")]))
        .collect();
    for req in &requests {
        let repr = futures::executor::block_on(kernel.issue(req.clone(), &root))
            .unwrap_or_else(|e| panic!("{}: {e}", req.target));
        assert_eq!(repr.expiry, Expiry::Never, "{}", req.target);
        let threads: Vec<String> = repr.threads().iter().map(|t| t.to_string()).collect();
        assert_eq!(threads, [CONFIG_THREAD], "{}", req.target);
        assert!(kernel.is_cached(req, &root), "{}", req.target);
    }
    kernel.cut(CONFIG_THREAD);
    for req in &requests {
        assert!(
            !kernel.is_cached(req, &root),
            "{}: one cut evicts every config-derived result",
            req.target
        );
    }
}

fn declared_outputs(kernel: &Kernel, iri: &str) -> Vec<String> {
    kernel
        .describe_pattern(iri)
        .unwrap_or_else(|| panic!("{iri} describes itself"))
        .outputs
        .iter()
        .map(|o| ikigai_conformance::rdf::bare_media_type(o))
        .collect()
}

/// What `ikigai-conformance` 0.1.0 does not check (its PENDING #11): a
/// declared output that is not an RDF face is never compared with what the
/// action serves. Every action, under each `as=` it takes: the media type
/// served is one the description declares, and every declared output is
/// served by some call.
#[test]
fn declared_outputs_are_the_media_types_served() {
    let stub = Stub::start();
    let kernel = kernel(discovering(&stub));
    let root = Capability::root();
    let json = ("as", "application/json");
    let calls: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("urn:llm:ask", vec![("prompt", PROMPT)]),
        ("urn:llm:ask", vec![("prompt", PROMPT), json]),
        ("urn:llm:ollama:ask", vec![("prompt", PROMPT)]),
        ("urn:llm:ollama:ask", vec![("prompt", PROMPT), json]),
        ("urn:llm:config", vec![]),
        ("urn:llm:models", vec![]),
        ("urn:llm:models", vec![("as", "text/turtle")]),
        ("urn:llm:select", vec![("needs", "cost=local")]),
        ("urn:llm:select", vec![("needs", "cost=local"), json]),
        ("urn:llm:ollama:up", vec![]),
        ("urn:llm:ollama:installed", vec![]),
        ("urn:llm:ollama:installed", vec![json]),
        ("urn:llm:ollama:model", vec![]),
        ("urn:llm:server:model", vec![]),
    ];
    let mut served_by_iri: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
    for (iri, args) in &calls {
        let repr =
            issue(&kernel, iri, args, &root).unwrap_or_else(|e| panic!("{iri} {args:?}: {e}"));
        let got = ikigai_conformance::rdf::bare_media_type(&repr.repr_type.media_type);
        let declared = declared_outputs(&kernel, iri);
        assert!(
            declared.contains(&got),
            "{iri} {args:?} served `{got}`, declared only {declared:?}"
        );
        served_by_iri.entry(iri).or_default().push(got);
    }
    for (iri, served) in served_by_iri {
        for face in declared_outputs(&kernel, iri) {
            assert!(
                served.contains(&face),
                "{iri} declares `{face}` but no call above served it"
            );
        }
    }
}

/// A prompt travels to the backend and nowhere else. The errors this module
/// composes — a capability denial, a bad `needs=`, a no-match, a discovery
/// failure, a backend 4xx — never carry it, and a server that echoes the
/// request beside its `error.message` (the `/echo` route) contributes only
/// the message. An error travels through traces, logs and MCP replies.
#[test]
fn a_prompt_never_appears_in_the_errors_this_module_composes() {
    let stub = Stub::start();
    let root = Capability::root();
    let mut registry = discovering(&stub);
    registry.providers.push(ollama(&stub, "/echo/v1"));
    registry.providers[2].provider = "echo".to_string();
    let mut absent = OpenAiConfig::discovering("absent", stub.base("/missing/v1"));
    absent.caps.vendor = Some("mlx".to_string());
    registry.providers.push(absent);
    let kernel = kernel(registry);
    let errors: Vec<(&str, Error)> = vec![
        (
            "denied",
            issue(
                &kernel,
                "urn:llm:ollama:ask",
                &[("prompt", PROMPT)],
                &scoped(&[OTHER_SCOPE]),
            )
            .unwrap_err(),
        ),
        (
            "bad needs",
            issue(
                &kernel,
                "urn:llm:ask",
                &[("prompt", PROMPT), ("needs", "speed>=9")],
                &root,
            )
            .unwrap_err(),
        ),
        (
            "no match",
            issue(
                &kernel,
                "urn:llm:ask",
                &[("prompt", PROMPT), ("needs", "audio")],
                &root,
            )
            .unwrap_err(),
        ),
        (
            "discovery failed",
            issue(&kernel, "urn:llm:absent:ask", &[("prompt", PROMPT)], &root).unwrap_err(),
        ),
        (
            "backend 4xx",
            issue(&kernel, "urn:llm:echo:ask", &[("prompt", PROMPT)], &root).unwrap_err(),
        ),
    ];
    for (case, err) in &errors {
        let text = format!("{err} / {err:?}");
        assert!(
            !text.contains(PROMPT),
            "{case}: the error carries the prompt: {text}"
        );
    }
    // The 4xx case is the one where a SERVER put the prompt in its body.
    let echoed = stub
        .received()
        .into_iter()
        .find(|r| r.path == "/echo/v1/chat/completions")
        .expect("the echo route was asked");
    assert!(String::from_utf8_lossy(&echoed.body).contains(PROMPT));
    let (_, backend_error) = &errors[4];
    assert!(
        backend_error.to_string().contains("400: bad request"),
        "only the server's error.message is kept: {backend_error}"
    );
    assert!(
        !backend_error.to_string().contains("echo"),
        "the fields beside error.message are dropped: {backend_error}"
    );
}

/// The contract as the manifold states it: one Source action per endpoint,
/// every input has a class, every network action declares exactly the net
/// wildcard and every config read declares nothing, and only `prompt` (the
/// asks) and `needs` (selection) are required.
#[test]
fn the_manifold_states_the_contract() {
    let stub = Stub::start();
    let kernel = kernel(discovering(&stub));
    for iri in NETWORK_ACTIONS.iter().chain(CONFIG_READS.iter()) {
        let description = kernel.describe_pattern(iri).unwrap();
        let specs = description.action_specs();
        assert_eq!(specs.len(), 1, "{iri}: one action");
        let spec = &specs[0];
        assert_eq!(spec.verb, Verb::Source, "{iri}");
        let expected: &[&str] = if NETWORK_ACTIONS.contains(iri) {
            &[NET_WILDCARD]
        } else {
            &[]
        };
        assert_eq!(spec.requires, expected, "{iri}: declared = enforced");
        for input in &spec.inputs {
            assert!(
                input.class.as_deref().is_some_and(|c| c.contains(':')),
                "{iri}: input `{}` has a class",
                input.name
            );
        }
        let required: Vec<&str> = spec
            .inputs
            .iter()
            .filter(|i| i.required)
            .map(|i| i.name.as_str())
            .collect();
        let expected: &[&str] = if iri.ends_with(":ask") {
            &["prompt"]
        } else if *iri == "urn:llm:select" {
            &["needs"]
        } else {
            &[]
        };
        assert_eq!(required, expected, "{iri}: the required inputs");
    }
}
