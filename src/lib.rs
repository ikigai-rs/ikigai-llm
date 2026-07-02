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

/// A backend's capability profile — the traits selection reasons over. All
/// **declared** (config-authored) for now; provider auto-discovery (e.g. Ollama's
/// `/api/show`) is a later slice that fills gaps, declared-wins.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Caps {
    /// Context window, in tokens.
    #[serde(default)]
    pub context: Option<u64>,
    /// Input modalities, e.g. `["text"]` or `["text", "vision"]`.
    #[serde(default)]
    pub modalities: Vec<String>,
    /// Supports tool / function calling.
    #[serde(default)]
    pub tools: Option<bool>,
    /// Supports a structured-output (JSON) mode.
    #[serde(default)]
    pub json: Option<bool>,
    /// Cost tier: `local` | `cheap` | `premium`.
    #[serde(default)]
    pub cost: Option<String>,
    /// Display-only parameter count, e.g. `"3B"`, `"70B"`.
    #[serde(default)]
    pub params: Option<String>,
}

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
    /// The declared capability profile (what `urn:llm:models` reports and
    /// selection will reason over). Empty by default.
    pub caps: Caps,
}

impl OpenAiConfig {
    /// A local Ollama on its OpenAI-compatible endpoint (no key needed).
    pub fn ollama(default_model: impl Into<String>) -> Self {
        OpenAiConfig {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
            default_model: default_model.into(),
            api_key: None,
            caps: Caps {
                cost: Some("local".to_string()),
                ..Caps::default()
            },
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
            #[serde(default)]
            caps: Caps,
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
                caps: e.caps,
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
        .bind(Exact::new("urn:llm:ask"), AskFacade::new(registry.clone()))
        .bind(
            Exact::new("urn:llm:config"),
            ConfigEndpoint::new(registry.clone()),
        )
        .bind(
            Exact::new("urn:llm:models"),
            ModelsEndpoint::new(registry.clone()),
        )
        .bind(
            Exact::new("urn:llm:select"),
            SelectEndpoint::new(registry.clone()),
        );
    for provider in &registry.providers {
        space = space
            .bind(
                Exact::new(format!("urn:llm:{}:ask", provider.provider)),
                OpenAiBackend::new(provider.clone(), Arc::clone(&transport)),
            )
            .bind(
                Exact::new(format!("urn:llm:{}:up", provider.provider)),
                UpEndpoint::new(provider.clone(), Arc::clone(&transport)),
            );
    }
    space
}

// ---- the facade -------------------------------------------------------------

/// `urn:llm:ask` — the front grammar. Picks a backend (`provider=` arg, else
/// `needs=` resolved over the trait profiles, else the configured default) and
/// re-issues the request to `urn:llm:<provider>:ask`.
pub struct AskFacade {
    registry: Registry,
}

impl AskFacade {
    /// A facade over `registry`: routes by `provider=`, else resolves `needs=`
    /// against the trait profiles, else falls back to the registry's default.
    pub fn new(registry: Registry) -> Self {
        AskFacade { registry }
    }
}

#[async_trait]
impl Endpoint for AskFacade {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        // Explicit `provider=` wins; then `needs=` (capability-based selection);
        // then the configured default.
        let provider = if let Ok(name) = inv.inline_str("provider") {
            name.to_string()
        } else if let Ok(needs) = inv.inline_str("needs") {
            select(&self.registry, &parse_needs(needs)?)
                .ok_or_else(|| no_match_error("urn:llm:ask", needs, &self.registry))?
                .provider
                .clone()
        } else {
            self.registry.default.clone()
        };
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
            "Ask an LLM: route to a backend (provider=, else needs= resolved over the \
             trait profiles, else the configured default) and return the completion.",
        )
        .input(
            ArgSpec::new("provider")
                .summary("backend to route to, e.g. ollama (default: configured)")
                .optional(),
        )
        .input(
            ArgSpec::new("needs")
                .summary("capability requirements, e.g. \"vision, ctx>=32k, cost<=cheap\"")
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
        let url = require_net(
            inv,
            &self.config.base_url,
            "chat/completions",
            &format!("urn:llm:{}:ask", self.config.provider),
        )?;

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

// ---- the model inventory ------------------------------------------------------

/// `urn:llm:models` — the annotated inventory: every configured backend with its
/// model and declared [`Caps`], as JSON (default) or Turtle (`as=text/turtle`).
/// The Turtle face is the queryable trait graph — SPARQL over it is how
/// capability-based selection ("a vision model with ≥32k context") will resolve.
/// Config-derived, so cacheable until the registry changes (a restart, for now).
pub struct ModelsEndpoint {
    registry: Registry,
}

impl ModelsEndpoint {
    fn new(registry: Registry) -> Self {
        ModelsEndpoint { registry }
    }

    fn as_json(&self) -> Value {
        let mut models = serde_json::Map::new();
        for p in &self.registry.providers {
            let caps = &p.caps;
            models.insert(
                p.provider.clone(),
                json!({
                    "backend": format!("urn:llm:{}:ask", p.provider),
                    "model": p.default_model,
                    "base_url": p.base_url,
                    "caps": {
                        "context": caps.context,
                        "modalities": caps.modalities,
                        "tools": caps.tools,
                        "json": caps.json,
                        "cost": caps.cost,
                        "params": caps.params,
                    },
                }),
            );
        }
        json!({ "default": self.registry.default, "models": models })
    }

    /// The trait graph. Vocabulary is module-local for now (`ik:LlmBackend`,
    /// `ik:model`, `ik:context`, `ik:modality`, `ik:tools`, `ik:jsonMode`,
    /// `ik:cost`, `ik:params`) — promotion into ikigai-vocab is a follow-up.
    fn as_turtle(&self) -> String {
        let mut ttl = String::from("@prefix ik: <https://ikigai-rs.dev/ns#> .\n");
        for p in &self.registry.providers {
            let mut props = vec![
                "a ik:LlmBackend".to_string(),
                format!("ik:model {}", ttl_str(&p.default_model)),
            ];
            if let Some(c) = p.caps.context {
                props.push(format!("ik:context {c}"));
            }
            for m in &p.caps.modalities {
                props.push(format!("ik:modality {}", ttl_str(m)));
            }
            if let Some(t) = p.caps.tools {
                props.push(format!("ik:tools {t}"));
            }
            if let Some(j) = p.caps.json {
                props.push(format!("ik:jsonMode {j}"));
            }
            if let Some(c) = &p.caps.cost {
                props.push(format!("ik:cost {}", ttl_str(c)));
            }
            if let Some(pr) = &p.caps.params {
                props.push(format!("ik:params {}", ttl_str(pr)));
            }
            ttl.push_str(&format!(
                "\n<urn:llm:{}:ask> {} .\n",
                p.provider,
                props.join(" ;\n    ")
            ));
        }
        ttl.push_str(&format!(
            "\n<urn:llm:ask> ik:routesTo <urn:llm:{}:ask> .\n",
            self.registry.default
        ));
        ttl
    }
}

/// A Turtle string literal (quote-and-escape).
fn ttl_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

#[async_trait]
impl Endpoint for ModelsEndpoint {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let want_turtle = inv
            .inline_str("as")
            .map(|s| s.contains("turtle"))
            .unwrap_or(false);
        let (media, bytes) = if want_turtle {
            ("text/turtle", self.as_turtle().into_bytes())
        } else {
            (
                "application/json",
                serde_json::to_vec(&self.as_json()).unwrap_or_default(),
            )
        };
        Ok(
            Representation::new(ReprType::new(media).with_param("charset", "utf-8"), bytes)
                .cacheable(),
        )
    }

    fn name(&self) -> &str {
        "llm-models"
    }

    fn describe(&self) -> Description {
        Description::new("urn:llm:models")
            .summary(
                "The annotated model inventory: every configured backend with its model \
                 and declared capability profile (context, modalities, tools, cost). \
                 JSON by default; `as=text/turtle` renders the queryable trait graph.",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .input(
                ArgSpec::new("as")
                    .summary("output representation: application/json (default) or text/turtle")
                    .optional(),
            )
            .output("application/json")
    }
}

// ---- capability-based selection -------------------------------------------------

/// One parsed requirement from a `needs=` expression.
#[derive(Debug, PartialEq)]
enum Need {
    /// `ctx>=32k` / `context>=32000` — context window at least this many tokens.
    ContextAtLeast(u64),
    /// `cost<=cheap` — cost tier at most this rank (local < cheap < premium).
    CostAtMost(u8),
    /// `cost=local` — exactly this cost tier.
    CostExactly(u8),
    /// `vision` / `modality=vision` — the modality must be declared.
    Modality(String),
    /// `tools` — declared tool/function calling.
    Tools,
    /// `json` — declared structured-output mode.
    Json,
}

/// The cost-tier ordering selection reasons over.
fn cost_rank(tier: &str) -> Option<u8> {
    match tier {
        "local" => Some(0),
        "cheap" => Some(1),
        "premium" => Some(2),
        _ => None,
    }
}

/// A token count, allowing a `k` suffix (`32k` = 32 × 1024).
fn parse_tokens(v: &str) -> Option<u64> {
    let v = v.trim();
    if let Some(n) = v.strip_suffix(['k', 'K']) {
        n.trim().parse::<u64>().ok().map(|n| n * 1024)
    } else {
        v.parse::<u64>().ok()
    }
}

/// Parse a comma-separated `needs=` expression: `trait`, `trait=v`, `trait>=v`,
/// `trait<=v`. Unknown terms are an error — a mistyped requirement must not
/// silently select the wrong model.
fn parse_needs(expr: &str) -> Result<Vec<Need>> {
    let bad = |term: &str, why: &str| Error::InvalidArgument {
        name: "needs".to_string(),
        detail: format!("`{term}`: {why}"),
    };
    let mut needs = Vec::new();
    for term in expr.split(',') {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        let need = if let Some((k, v)) = term.split_once(">=") {
            match k.trim() {
                "ctx" | "context" => Need::ContextAtLeast(
                    parse_tokens(v).ok_or_else(|| bad(term, "expected a token count"))?,
                ),
                other => return Err(bad(term, &format!("`{other}` does not support >="))),
            }
        } else if let Some((k, v)) = term.split_once("<=") {
            match k.trim() {
                "cost" => Need::CostAtMost(
                    cost_rank(v.trim()).ok_or_else(|| bad(term, "expected local|cheap|premium"))?,
                ),
                other => return Err(bad(term, &format!("`{other}` does not support <="))),
            }
        } else if let Some((k, v)) = term.split_once('=') {
            match k.trim() {
                "cost" => Need::CostExactly(
                    cost_rank(v.trim()).ok_or_else(|| bad(term, "expected local|cheap|premium"))?,
                ),
                "modality" => Need::Modality(v.trim().to_string()),
                other => return Err(bad(term, &format!("unknown trait `{other}`"))),
            }
        } else {
            match term {
                "tools" => Need::Tools,
                "json" => Need::Json,
                "text" | "vision" | "audio" => Need::Modality(term.to_string()),
                other => return Err(bad(term, &format!("unknown requirement `{other}`"))),
            }
        };
        needs.push(need);
    }
    Ok(needs)
}

/// Does a declared profile satisfy every need? Conservative: a trait the
/// provider didn't declare cannot satisfy a requirement on it.
fn caps_satisfy(caps: &Caps, needs: &[Need]) -> bool {
    needs.iter().all(|need| match need {
        Need::ContextAtLeast(n) => caps.context.is_some_and(|c| c >= *n),
        Need::CostAtMost(r) => caps
            .cost
            .as_deref()
            .and_then(cost_rank)
            .is_some_and(|c| c <= *r),
        Need::CostExactly(r) => caps.cost.as_deref().and_then(cost_rank) == Some(*r),
        Need::Modality(m) => caps.modalities.iter().any(|x| x == m),
        Need::Tools => caps.tools == Some(true),
        Need::Json => caps.json == Some(true),
    })
}

/// The satisfying provider under the policy **cheapest-that-fits → smallest
/// context → registry order** (an undeclared cost tier sorts last).
fn select<'a>(registry: &'a Registry, needs: &[Need]) -> Option<&'a OpenAiConfig> {
    registry
        .providers
        .iter()
        .enumerate()
        .filter(|(_, p)| caps_satisfy(&p.caps, needs))
        .min_by_key(|(i, p)| {
            (
                p.caps.cost.as_deref().and_then(cost_rank).unwrap_or(3),
                p.caps.context.unwrap_or(u64::MAX),
                *i,
            )
        })
        .map(|(_, p)| p)
}

/// A no-match error that says what was asked and what was available.
fn no_match_error(who: &str, needs: &str, registry: &Registry) -> Error {
    let available: Vec<&str> = registry
        .providers
        .iter()
        .map(|p| p.provider.as_str())
        .collect();
    Error::Endpoint(format!(
        "{who}: no configured backend satisfies `{needs}` (providers: {}) — see urn:llm:models",
        available.join(", ")
    ))
}

/// `urn:llm:select` — deterministic capability-based selection: resolve a
/// `needs=` expression over the declared trait profiles and return the winning
/// backend's IRI (`text/plain`, pipeable) or its detail (`as=application/json`).
/// Selection is a pure function of the registry, so the result is cacheable.
/// (The SPARQL power path is composition, not a dep: `urn:llm:models
/// as=text/turtle` is the same trait data as a queryable graph.)
pub struct SelectEndpoint {
    registry: Registry,
}

impl SelectEndpoint {
    fn new(registry: Registry) -> Self {
        SelectEndpoint { registry }
    }
}

#[async_trait]
impl Endpoint for SelectEndpoint {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let needs = inv.inline_str("needs")?;
        let winner = select(&self.registry, &parse_needs(needs)?)
            .ok_or_else(|| no_match_error("urn:llm:select", needs, &self.registry))?;
        let want_json = inv
            .inline_str("as")
            .map(|s| s.contains("json"))
            .unwrap_or(false);
        let (media, bytes) = if want_json {
            let detail = json!({
                "backend": format!("urn:llm:{}:ask", winner.provider),
                "provider": winner.provider,
                "model": winner.default_model,
                "cost": winner.caps.cost,
                "context": winner.caps.context,
            });
            (
                "application/json",
                serde_json::to_vec(&detail).unwrap_or_default(),
            )
        } else {
            (
                "text/plain",
                format!("urn:llm:{}:ask", winner.provider).into_bytes(),
            )
        };
        Ok(
            Representation::new(ReprType::new(media).with_param("charset", "utf-8"), bytes)
                .cacheable(),
        )
    }

    fn name(&self) -> &str {
        "llm-select"
    }

    fn describe(&self) -> Description {
        Description::new("urn:llm:select")
            .summary(
                "Capability-based selection: resolve `needs` (e.g. \"vision, ctx>=32k, \
                 cost<=cheap\") over the declared trait profiles and return the winning \
                 backend IRI. Policy: cheapest-that-fits, then smallest context, then \
                 registry order.",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .input(ArgSpec::new("needs").summary(
                "comma-separated requirements: ctx>=N[k] · cost<=tier · cost=tier · \
                 modality=X (or bare text/vision/audio) · tools · json",
            ))
            .input(
                ArgSpec::new("as")
                    .summary("text/plain backend IRI (default) or application/json detail")
                    .optional(),
            )
            .output("text/plain;charset=utf-8")
    }
}

// ---- liveness -----------------------------------------------------------------

/// `urn:llm:<provider>:up` — a boolean liveness resource: `true` if the provider
/// answers a cheap `GET {base_url}/models`, else `false`. Made for
/// `urn:fn:conditional` (`if=urn:llm:ollama:up then=<demo> else=<offline note>`),
/// so LLM demos degrade gracefully instead of erroring when the server is down.
/// Deliberately **uncacheable** — liveness is a live fact. A capability that
/// cannot reach the host is an error, not `false` (denied ≠ down).
pub struct UpEndpoint {
    config: OpenAiConfig,
    transport: Arc<dyn HttpTransport>,
}

impl UpEndpoint {
    fn new(config: OpenAiConfig, transport: Arc<dyn HttpTransport>) -> Self {
        UpEndpoint { config, transport }
    }
}

#[async_trait]
impl Endpoint for UpEndpoint {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let url = require_net(
            inv,
            &self.config.base_url,
            "models",
            &format!("urn:llm:{}:up", self.config.provider),
        )?;
        let alive = match self
            .transport
            .send(HttpRequest {
                method: Method::Get,
                url,
                headers: Vec::new(),
                body: Vec::new(),
            })
            .await
        {
            Ok(response) => response.status < 400,
            Err(_) => false, // unreachable server = down, not an error
        };
        Ok(Representation::new(
            ReprType::new("text/plain").with_param("charset", "utf-8"),
            if alive {
                b"true".to_vec()
            } else {
                b"false".to_vec()
            },
        ))
    }

    fn name(&self) -> &str {
        "llm-up"
    }

    fn describe(&self) -> Description {
        Description::new(format!("urn:llm:{}:up", self.config.provider))
            .summary(
                "Boolean liveness: `true` if the provider answers a cheap GET, else \
                 `false`. Branch on it with urn:fn:conditional to degrade gracefully.",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .output("text/plain;charset=utf-8")
            .requires(CAP_NET)
    }
}

/// Gate an outbound call by the per-host net capability and return the full URL:
/// `{base_url}/{path}` checked via [`ikigai_http::net_allows`]. Shared by every
/// endpoint here that touches the provider.
fn require_net(inv: &Invocation<'_>, base_url: &str, path: &str, who: &str) -> Result<String> {
    let url = format!("{}/{path}", base_url.trim_end_matches('/'));
    let parsed = url::Url::parse(&url)
        .map_err(|e| Error::Endpoint(format!("{who}: bad base_url `{base_url}`: {e}")))?;
    let host = parsed.host_str().unwrap_or("");
    if !ikigai_http::net_allows(inv.capability, host, parsed.path()) {
        return Err(Error::Endpoint(format!(
            "{who}: capability does not allow reaching `{host}` (needs urn:cap:net:{host})"
        )));
    }
    Ok(url)
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
            "fast":   { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
                        "caps": { "context": 131072, "modalities": ["text"], "tools": true,
                                  "cost": "local", "params": "3B" } },
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
    fn from_json_parses_caps() {
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let fast = reg.providers.iter().find(|p| p.provider == "fast").unwrap();
        assert_eq!(fast.caps.context, Some(131072));
        assert_eq!(fast.caps.tools, Some(true));
        assert_eq!(fast.caps.cost.as_deref(), Some("local"));
        // caps are optional — `remote` declared none.
        let remote = reg
            .providers
            .iter()
            .find(|p| p.provider == "remote")
            .unwrap();
        assert_eq!(remote.caps.context, None);
    }

    fn issue(kernel: &Kernel, req: Request) -> Representation {
        block_on(kernel.issue(req, &Capability::root())).unwrap()
    }

    #[test]
    fn models_reports_the_annotated_inventory() {
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(CANNED), reg)));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap()),
        );
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        assert_eq!(v["default"], "fast");
        assert_eq!(v["models"]["fast"]["backend"], "urn:llm:fast:ask");
        assert_eq!(v["models"]["fast"]["caps"]["context"], 131072);
        assert_eq!(v["models"]["remote"]["model"], "gpt-4o");
    }

    #[test]
    fn models_renders_the_trait_graph_as_turtle() {
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(CANNED), reg)));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap())
                .with_arg("as", ArgRef::Inline(b"text/turtle".to_vec())),
        );
        let ttl = String::from_utf8(out.bytes).unwrap();
        assert!(ttl.contains("<urn:llm:fast:ask> a ik:LlmBackend"));
        assert!(ttl.contains("ik:context 131072"));
        assert!(ttl.contains("ik:tools true"));
        assert!(ttl.contains("<urn:llm:ask> ik:routesTo <urn:llm:fast:ask>"));
    }

    /// Three providers with distinct trait profiles, for selection tests:
    /// fast = local text 128k · seer = local vision 32k · posh = premium vision+tools 128k.
    const SELECTABLE: &str = r#"{
        "default": "fast",
        "providers": {
            "fast": { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
                      "caps": { "context": 131072, "modalities": ["text"], "cost": "local" } },
            "seer": { "base_url": "http://localhost:11434/v1", "model": "llava:7b",
                      "caps": { "context": 32768, "modalities": ["text", "vision"], "cost": "local" } },
            "posh": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "sk-S",
                      "caps": { "context": 131072, "modalities": ["text", "vision"], "tools": true, "cost": "premium" } }
        }
    }"#;

    fn select_kernel(mock: Arc<MockTransport>) -> Kernel {
        let reg = Registry::from_json(SELECTABLE).unwrap();
        Kernel::new(Arc::new(space(mock, reg)))
    }

    fn selected(kernel: &Kernel, needs: &str) -> String {
        let out = issue(
            kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                .with_arg("needs", ArgRef::Inline(needs.as_bytes().to_vec())),
        );
        String::from_utf8(out.bytes).unwrap()
    }

    #[test]
    fn select_picks_the_cheapest_that_fits() {
        let kernel = select_kernel(MockTransport::new(CANNED));
        // seer and posh both see; seer is local (cheaper) -> wins.
        assert_eq!(selected(&kernel, "vision"), "urn:llm:seer:ask");
        // only posh has tools + vision.
        assert_eq!(selected(&kernel, "vision, tools"), "urn:llm:posh:ask");
        // fast and posh have >=100k context; fast is local -> wins. (32k = 32768.)
        assert_eq!(selected(&kernel, "ctx>=100k"), "urn:llm:fast:ask");
        // seer fits 32k exactly and ties with fast on cost; smaller context wins.
        assert_eq!(
            selected(&kernel, "ctx>=32k, cost<=cheap"),
            "urn:llm:seer:ask"
        );
    }

    #[test]
    fn select_reports_no_match_and_bad_terms_clearly() {
        let kernel = select_kernel(MockTransport::new(CANNED));
        let no_match = block_on(
            kernel.issue(
                Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                    .with_arg("needs", ArgRef::Inline(b"audio".to_vec())),
                &Capability::root(),
            ),
        );
        let msg = format!("{:?}", no_match.unwrap_err());
        assert!(msg.contains("no configured backend satisfies"), "{msg}");
        assert!(msg.contains("fast"), "lists providers: {msg}");

        let bad_term = block_on(
            kernel.issue(
                Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                    .with_arg("needs", ArgRef::Inline(b"speed>=9".to_vec())),
                &Capability::root(),
            ),
        );
        assert!(
            bad_term.is_err(),
            "unknown trait must error, not mis-select"
        );
    }

    #[test]
    fn select_as_json_returns_the_detail() {
        let kernel = select_kernel(MockTransport::new(CANNED));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                .with_arg("needs", ArgRef::Inline(b"vision".to_vec()))
                .with_arg("as", ArgRef::Inline(b"application/json".to_vec())),
        );
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        assert_eq!(v["backend"], "urn:llm:seer:ask");
        assert_eq!(v["model"], "llava:7b");
    }

    #[test]
    fn the_facade_routes_by_needs() {
        let mock = MockTransport::new(CANNED);
        let kernel = select_kernel(Arc::clone(&mock));
        let out = issue(
            &kernel,
            ask("urn:llm:ask", "hi").with_arg("needs", ArgRef::Inline(b"vision".to_vec())),
        );
        assert_eq!(out.bytes, b"Hello there!");
        // the request went to seer's backend: the posted payload names its model.
        let sent = mock.last.lock().unwrap().take().unwrap();
        let body = String::from_utf8(sent.body).unwrap();
        assert!(body.contains("llava:7b"), "routed to seer: {body}");
    }

    /// A transport whose server is unreachable — every send fails.
    struct DownTransport;

    #[async_trait]
    impl HttpTransport for DownTransport {
        async fn send(&self, _r: HttpRequest) -> std::result::Result<HttpResponse, String> {
            Err("connection refused".to_string())
        }
    }

    #[test]
    fn up_is_true_when_the_provider_answers() {
        let mock = MockTransport::new("{}");
        let kernel = kernel_with(Arc::clone(&mock));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:ollama:up").unwrap()),
        );
        assert_eq!(out.bytes, b"true");
        // and the ping was the cheap models listing, not a completion
        let pinged = mock.last.lock().unwrap().clone().unwrap();
        assert_eq!(pinged.url, "http://localhost:11434/v1/models");
        assert_eq!(pinged.method, Method::Get);
    }

    #[test]
    fn up_is_false_when_the_provider_is_unreachable() {
        let kernel = Kernel::new(Arc::new(space(
            Arc::new(DownTransport),
            OpenAiConfig::ollama("llama3.1"),
        )));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:ollama:up").unwrap()),
        );
        assert_eq!(out.bytes, b"false");
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
