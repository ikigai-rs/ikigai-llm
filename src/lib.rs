//! `ikigai-llm` — flexible LLM inference as ROC resources.
//!
//! One **front-facing grammar** (`urn:llm:ask`) backed by pluggable **backend
//! modules**, each also directly addressable (`urn:llm:<provider>:ask`). The
//! facade resolves *which* backend to use (a `provider=` arg, else the configured
//! default) and **re-issues** the request to that backend through the kernel — so
//! the backend's cache validity and golden threads propagate to the facade's
//! result for free.
//!
//! This is the imperative seed of the interception/rewrite keystone: a static
//! alias is a [`Rewrite`](ikigai_core::Rewrite) space, but selection that must
//! read arguments or config is expressed as an endpoint whose `invoke` does the
//! rewrite. A good `AskFacade` is a candidate to generalize into the declarative
//! overlay primitive later.
//!
//! ## Slice 0
//! A single **OpenAI-compatible chat backend** over REST. That one shape covers
//! Ollama, vLLM, `llama.cpp`'s server, `mlx_lm.server`, and LM Studio — they
//! differ only in `base_url`, `default_model`, and whether a key is needed. The
//! facade already dispatches by provider, so more backends slot in behind the
//! same front grammar without changing it. Buffered (no streaming yet),
//! single-turn, `urn:cap:net`-gated. Generation is non-deterministic, so results
//! are uncacheable by default (deterministic caching is a later slice).

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use ikigai_core::{
    ArgSpec, Description, Endpoint, EndpointSpace, Error, Exact, Invocation, Iri, ReprType,
    Representation, Request, Result, Verb,
};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport, Method};
use serde::Deserialize;
use serde_json::{json, Value};

/// The capability every backend call requires — reaching a server, even on
/// localhost, is a network act.
const CAP_NET: &str = "urn:cap:net";

/// Configuration for an OpenAI-compatible chat backend. One shape covers Ollama,
/// vLLM, `llama.cpp`'s server, `mlx_lm.server`, and LM Studio.
#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    /// The IRI segment this backend binds under (`urn:llm:<provider>:ask`) and the
    /// value the facade defaults to. e.g. `"ollama"`.
    pub provider: String,
    /// Base URL up to (not including) `/chat/completions`,
    /// e.g. `"http://localhost:11434/v1"`.
    pub base_url: String,
    /// Model used when a request doesn't name one.
    pub default_model: String,
    /// Bearer token, if the endpoint needs one. Local runtimes usually don't.
    pub api_key: Option<String>,
}

impl OpenAiConfig {
    /// A local Ollama on its OpenAI-compatible endpoint (no key needed).
    pub fn ollama(default_model: impl Into<String>) -> Self {
        OpenAiConfig {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
            default_model: default_model.into(),
            api_key: None,
        }
    }
}

/// A set of configured providers plus the default the facade routes to when a
/// request names none. Built from compiled defaults ⊕ an optional hand-editable
/// file — the load-time form of the "logical URI aliases to file-or-code" pattern.
#[derive(Clone, Debug)]
pub struct Registry {
    /// The provider `urn:llm:ask` routes to when a request names none.
    pub default: String,
    /// Every configured backend; each is bound at `urn:llm:<provider>:ask`.
    pub providers: Vec<OpenAiConfig>,
}

impl Registry {
    /// A single-provider registry (that provider becomes the default).
    pub fn single(config: OpenAiConfig) -> Self {
        Registry {
            default: config.provider.clone(),
            providers: vec![config],
        }
    }

    /// Parse a registry from JSON (the hand-editable config file):
    ///
    /// ```json
    /// { "default": "fast",
    ///   "providers": {
    ///     "fast":   { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b" },
    ///     "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "…" }
    ///   } }
    /// ```
    pub fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Entry {
            base_url: String,
            #[serde(alias = "default_model")]
            model: String,
            #[serde(default)]
            api_key: Option<String>,
        }
        #[derive(Deserialize)]
        struct Doc {
            default: String,
            providers: BTreeMap<String, Entry>,
        }
        let doc: Doc = serde_json::from_str(json)
            .map_err(|e| Error::Endpoint(format!("urn:llm:config: invalid JSON: {e}")))?;
        let providers = doc
            .providers
            .into_iter()
            .map(|(name, e)| OpenAiConfig {
                provider: name,
                base_url: e.base_url,
                default_model: e.model,
                api_key: e.api_key,
            })
            .collect();
        Ok(Registry {
            default: doc.default,
            providers,
        })
    }
}

impl Default for Registry {
    fn default() -> Self {
        Registry::single(OpenAiConfig::ollama("llama3.2"))
    }
}

impl From<OpenAiConfig> for Registry {
    fn from(config: OpenAiConfig) -> Self {
        Registry::single(config)
    }
}

/// Mount the LLM facade, one backend per configured provider, and the config
/// resource.
///
/// Binds `urn:llm:ask` (the facade), a `urn:llm:<provider>:ask` for every provider
/// in the registry (each directly addressable and catalog-advertised with its own
/// model/base_url), and `urn:llm:config` (the effective registry, keys redacted).
/// Accepts a [`Registry`] or — via `From<OpenAiConfig>` — a single [`OpenAiConfig`].
/// The host supplies the [`HttpTransport`].
pub fn space(transport: Arc<dyn HttpTransport>, registry: impl Into<Registry>) -> EndpointSpace {
    let registry = registry.into();
    let mut space = EndpointSpace::new()
        .bind(Exact::new("urn:llm:ask"), AskFacade::new(&registry.default))
        .bind(
            Exact::new("urn:llm:config"),
            ConfigEndpoint::new(registry.clone()),
        );
    for provider in &registry.providers {
        let iri = format!("urn:llm:{}:ask", provider.provider);
        space = space.bind(
            Exact::new(iri),
            OpenAiBackend::new(provider.clone(), Arc::clone(&transport)),
        );
    }
    space
}

// ---- the facade -------------------------------------------------------------

/// `urn:llm:ask` — the front grammar. Picks a backend (`provider=` arg, else the
/// configured default) and re-issues the request to `urn:llm:<provider>:ask`.
pub struct AskFacade {
    default_provider: String,
}

impl AskFacade {
    /// A facade defaulting to `default_provider` when a request names none.
    pub fn new(default_provider: impl Into<String>) -> Self {
        AskFacade {
            default_provider: default_provider.into(),
        }
    }
}

#[async_trait]
impl Endpoint for AskFacade {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let provider = inv
            .inline_str("provider")
            .map(str::to_string)
            .unwrap_or_else(|_| self.default_provider.clone());
        let target = Iri::parse(format!("urn:llm:{provider}:ask"))
            .map_err(|e| Error::Endpoint(format!("urn:llm:ask: bad provider `{provider}`: {e}")))?;
        // Rewrite the target; carry every argument through unchanged. Going via
        // `inv.issue` records the backend result as a dependency, so its expiry
        // and golden threads propagate to this facade result.
        let mut req = Request::new(Verb::Source, target);
        req.args = inv.request.args.clone();
        inv.issue(req).await
    }

    fn name(&self) -> &str {
        "llm-ask"
    }

    fn describe(&self) -> Description {
        ask_description(
            "urn:llm:ask",
            "Ask an LLM: route to a backend (provider= or the configured default) and return the completion.",
        )
        .input(
            ArgSpec::new("provider")
                .summary("backend to route to, e.g. ollama (default: configured)")
                .optional(),
        )
    }
}

// ---- the OpenAI-compatible backend -----------------------------------------

/// `urn:llm:<provider>:ask` — one OpenAI-compatible chat backend over REST.
pub struct OpenAiBackend {
    config: OpenAiConfig,
    transport: Arc<dyn HttpTransport>,
}

impl OpenAiBackend {
    /// A backend for `config`, sending over `transport`.
    pub fn new(config: OpenAiConfig, transport: Arc<dyn HttpTransport>) -> Self {
        OpenAiBackend { config, transport }
    }
}

#[async_trait]
impl Endpoint for OpenAiBackend {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        // Reaching a server — even localhost — is a network act, gated per-host by
        // the same capability convention ikigai-http uses (urn:cap:net:<host>).
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let parsed = url::Url::parse(&url).map_err(|e| {
            Error::Endpoint(format!("llm: bad base_url `{}`: {e}", self.config.base_url))
        })?;
        let host = parsed.host_str().unwrap_or("");
        if !ikigai_http::net_allows(inv.capability, host, parsed.path()) {
            return Err(Error::Endpoint(format!(
                "urn:llm:{}:ask: capability does not allow reaching `{host}` (needs urn:cap:net:{host})",
                self.config.provider
            )));
        }

        // Prompt: explicit `prompt=`, else the piped `content`.
        let prompt = inv
            .inline_str("prompt")
            .or_else(|_| inv.inline_str("content"))
            .map_err(|_| Error::MissingArgument("prompt".to_string()))?;
        let model = inv
            .inline_str("model")
            .map(str::to_string)
            .unwrap_or_else(|_| self.config.default_model.clone());

        let mut messages = Vec::new();
        if let Ok(system) = inv.inline_str("system") {
            messages.push(json!({ "role": "system", "content": system }));
        }
        messages.push(json!({ "role": "user", "content": prompt }));

        let mut payload = json!({ "model": model, "messages": messages, "stream": false });
        if let Ok(t) = inv.inline_str("temperature").and_then(parse_f64) {
            payload["temperature"] = json!(t);
        }
        if let Ok(m) = inv.inline_str("max_tokens").and_then(parse_u64) {
            payload["max_tokens"] = json!(m);
        }

        let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
        if let Some(key) = &self.config.api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }

        let body = serde_json::to_vec(&payload)
            .map_err(|e| Error::Endpoint(format!("llm: encoding request failed: {e}")))?;
        let response: HttpResponse = self
            .transport
            .send(HttpRequest {
                method: Method::Post,
                url,
                headers,
                body,
            })
            .await
            .map_err(|e| Error::Endpoint(format!("llm transport error: {e}")))?;

        if response.status >= 400 {
            let detail = String::from_utf8_lossy(&response.body);
            return Err(Error::Endpoint(format!(
                "llm backend returned {}: {detail}",
                response.status
            )));
        }

        let parsed: Value = serde_json::from_slice(&response.body)
            .map_err(|e| Error::Endpoint(format!("llm: response was not JSON: {e}")))?;
        let choice = &parsed["choices"][0];
        let text = choice["message"]["content"].as_str().ok_or_else(|| {
            Error::Endpoint("llm: no choices[0].message.content in response".to_string())
        })?;

        // Default: the completion text. `as=application/json`: a normalized
        // envelope. Uncacheable by default (generation is non-deterministic).
        let want_json = inv
            .inline_str("as")
            .map(|s| s.contains("json"))
            .unwrap_or(false);
        if want_json {
            let envelope = json!({
                "text": text,
                "model": parsed["model"].as_str().unwrap_or(model.as_str()),
                "finish_reason": choice["finish_reason"].as_str(),
                "usage": parsed.get("usage").cloned().unwrap_or(Value::Null),
            });
            Ok(Representation::new(
                ReprType::new("application/json").with_param("charset", "utf-8"),
                serde_json::to_vec(&envelope).unwrap_or_default(),
            ))
        } else {
            Ok(Representation::new(
                ReprType::new("text/plain").with_param("charset", "utf-8"),
                text.as_bytes().to_vec(),
            ))
        }
    }

    fn name(&self) -> &str {
        "llm-openai"
    }

    fn describe(&self) -> Description {
        ask_description(
            &format!("urn:llm:{}:ask", self.config.provider),
            "Chat completion via an OpenAI-compatible backend (Ollama/vLLM/llama.cpp/mlx_lm/LM Studio).",
        )
    }
}

/// `Ok(v)` when `s` parses as an `f64`, else a throwaway error (so callers use
/// `.and_then(parse_f64)` and ignore unparseable values).
fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>().map_err(|_| Error::InvalidArgument {
        name: "temperature".to_string(),
        detail: "expected a number".to_string(),
    })
}

fn parse_u64(s: &str) -> Result<u64> {
    s.parse::<u64>().map_err(|_| Error::InvalidArgument {
        name: "max_tokens".to_string(),
        detail: "expected an integer".to_string(),
    })
}

/// The shared parameter contract for the facade and the backends.
fn ask_description(id: &str, summary: &str) -> Description {
    Description::new(id)
        .summary(summary)
        .verb(Verb::Source)
        .input(ArgSpec::new("prompt").summary("the user prompt (or pipe it in as content)"))
        .input(
            ArgSpec::new("model")
                .summary("model name (default: the backend's)")
                .optional(),
        )
        .input(ArgSpec::new("system").summary("system prompt").optional())
        .input(
            ArgSpec::new("temperature")
                .summary("sampling temperature")
                .optional(),
        )
        .input(
            ArgSpec::new("max_tokens")
                .summary("maximum tokens to generate")
                .optional(),
        )
        .input(
            ArgSpec::new("as")
                .summary("application/json for a {text,model,usage} envelope; default text/plain")
                .optional(),
        )
        .output("text/plain;charset=utf-8")
        .requires(CAP_NET)
}

// ---- the config resource ----------------------------------------------------

/// `urn:llm:config` — reports the effective registry (default + configured
/// providers), with API keys **redacted** so the resource never leaks a secret.
pub struct ConfigEndpoint {
    registry: Registry,
}

impl ConfigEndpoint {
    fn new(registry: Registry) -> Self {
        ConfigEndpoint { registry }
    }
}

#[async_trait]
impl Endpoint for ConfigEndpoint {
    async fn invoke(&self, _inv: &Invocation<'_>) -> Result<Representation> {
        let mut providers = serde_json::Map::new();
        for p in &self.registry.providers {
            providers.insert(
                p.provider.clone(),
                json!({
                    "base_url": p.base_url,
                    "model": p.default_model,
                    "api_key": p.api_key.as_ref().map(|_| "***"),
                }),
            );
        }
        let out = json!({ "default": self.registry.default, "providers": providers });
        Ok(Representation::new(
            ReprType::new("application/json").with_param("charset", "utf-8"),
            serde_json::to_vec(&out).unwrap_or_default(),
        )
        .cacheable())
    }

    fn name(&self) -> &str {
        "llm-config"
    }

    fn describe(&self) -> Description {
        Description::new("urn:llm:config")
            .summary(
                "The effective LLM provider registry (default + configured backends; \
                 API keys redacted).",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .output("application/json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ikigai_core::{ArgRef, Capability, Kernel, Space};
    use std::sync::Mutex;

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        futures::executor::block_on(f)
    }

    /// A canned OpenAI-compatible response; records the request it was sent.
    struct MockTransport {
        response: HttpResponse,
        last: Mutex<Option<HttpRequest>>,
    }

    impl MockTransport {
        fn new(body: &str) -> Arc<Self> {
            Arc::new(MockTransport {
                response: HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: body.as_bytes().to_vec(),
                },
                last: Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl HttpTransport for MockTransport {
        async fn send(&self, request: HttpRequest) -> std::result::Result<HttpResponse, String> {
            *self.last.lock().unwrap() = Some(request);
            Ok(self.response.clone())
        }
    }

    const CANNED: &str = r#"{"model":"llama3.1","choices":[{"message":{"role":"assistant","content":"Hello there!"},"finish_reason":"stop"}],"usage":{"total_tokens":7}}"#;

    fn kernel_with(mock: Arc<MockTransport>) -> Kernel {
        Kernel::new(Arc::new(space(mock, OpenAiConfig::ollama("llama3.1"))))
    }

    fn ask(iri: &str, prompt: &str) -> Request {
        Request::new(Verb::Source, Iri::parse(iri).unwrap())
            .with_arg("prompt", ArgRef::Inline(prompt.as_bytes().to_vec()))
    }

    #[test]
    fn backend_posts_chat_completions_and_returns_text() {
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(Arc::clone(&mock));
        let out =
            block_on(kernel.issue(ask("urn:llm:ollama:ask", "hi"), &Capability::root())).unwrap();
        assert_eq!(out.bytes, b"Hello there!");

        let sent = mock.last.lock().unwrap().clone().unwrap();
        assert_eq!(sent.method, Method::Post);
        assert_eq!(sent.url, "http://localhost:11434/v1/chat/completions");
        let body: Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["model"], "llama3.1");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
    }

    #[test]
    fn facade_routes_to_the_default_provider_backend() {
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(Arc::clone(&mock));
        // urn:llm:ask (no provider=) must re-issue to urn:llm:ollama:ask.
        let out = block_on(kernel.issue(ask("urn:llm:ask", "hi"), &Capability::root())).unwrap();
        assert_eq!(out.bytes, b"Hello there!");
        assert!(
            mock.last.lock().unwrap().is_some(),
            "the backend was reached"
        );
    }

    #[test]
    fn system_prompt_is_prepended() {
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(Arc::clone(&mock));
        let req = ask("urn:llm:ollama:ask", "hi")
            .with_arg("system", ArgRef::Inline(b"be terse".to_vec()));
        block_on(kernel.issue(req, &Capability::root())).unwrap();
        let sent = mock.last.lock().unwrap().clone().unwrap();
        let body: Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be terse");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn json_envelope_when_requested() {
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(mock);
        let req = ask("urn:llm:ollama:ask", "hi")
            .with_arg("as", ArgRef::Inline(b"application/json".to_vec()));
        let out = block_on(kernel.issue(req, &Capability::root())).unwrap();
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        assert_eq!(v["text"], "Hello there!");
        assert_eq!(v["model"], "llama3.1");
        assert_eq!(v["finish_reason"], "stop");
    }

    #[test]
    fn network_capability_is_required() {
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(mock);
        let none = Capability::root().attenuate(Vec::<String>::new());
        let denied = block_on(kernel.issue(ask("urn:llm:ollama:ask", "hi"), &none));
        assert!(denied.is_err(), "no net capability -> denied");
    }

    #[test]
    fn a_host_scoped_capability_authorizes_that_host() {
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(mock);
        // Ollama's default base_url is http://localhost:11434/v1 -> host "localhost".
        let scoped = Capability::root().attenuate(["urn:cap:net:localhost".to_string()]);
        let out = block_on(kernel.issue(ask("urn:llm:ollama:ask", "hi"), &scoped)).unwrap();
        assert_eq!(out.bytes, b"Hello there!");
    }

    const TWO_PROVIDERS: &str = r#"{
        "default": "fast",
        "providers": {
            "fast":   { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b" },
            "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "sk-SECRET" }
        }
    }"#;

    #[test]
    fn from_json_parses_a_multi_provider_registry() {
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        assert_eq!(reg.default, "fast");
        assert_eq!(reg.providers.len(), 2);
        let remote = reg
            .providers
            .iter()
            .find(|p| p.provider == "remote")
            .unwrap();
        assert_eq!(remote.default_model, "gpt-4o");
        assert_eq!(remote.api_key.as_deref(), Some("sk-SECRET"));
    }

    #[test]
    fn space_advertises_one_backend_per_provider() {
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let patterns: Vec<String> = space(MockTransport::new(CANNED), reg)
            .entries()
            .unwrap()
            .into_iter()
            .map(|e| e.pattern)
            .collect();
        for expected in [
            "urn:llm:ask",
            "urn:llm:config",
            "urn:llm:fast:ask",
            "urn:llm:remote:ask",
        ] {
            assert!(patterns.iter().any(|p| p == expected), "missing {expected}");
        }
    }

    #[test]
    fn config_resource_reports_providers_with_keys_redacted() {
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(CANNED), reg)));
        let out = block_on(kernel.issue(
            Request::new(Verb::Source, Iri::parse("urn:llm:config").unwrap()),
            &Capability::root(),
        ))
        .unwrap();
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        assert_eq!(v["default"], "fast");
        assert_eq!(v["providers"]["remote"]["api_key"], "***");
        assert_eq!(v["providers"]["fast"]["api_key"], Value::Null);
        assert!(
            !String::from_utf8_lossy(&out.bytes).contains("sk-SECRET"),
            "the real key must never appear"
        );
    }
}
