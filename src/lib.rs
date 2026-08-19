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
/// localhost, is a network act. Declared in the **wildcard form** (the
/// constitution's rule for parameterized ACL families): the manifold offers the
/// action to any capability holding SOME `urn:cap:net:*` grant, and the kernel's
/// baseline `requires` enforcement (core ≥ 0.1.49) passes it on the same
/// predicate; `require_net` then checks the actual host against the grant — the
/// declared floor, the per-host ceiling.
const CAP_NET: &str = "urn:cap:net:*";

/// A backend's capability profile — the traits selection reasons over. Facts
/// arrive at three strengths: **annotations** (an alignment graph, authoritative
/// — may override, loudly) > **declared** (config-authored) > **discovered**
/// (Ollama's `/api/show` and an OpenAI-compat `/v1/models` listing, fills gaps
/// only).
///
/// **Two kinds of fact live here and they are sourced differently.** `context`,
/// `modalities`, `tools` and `json` are *capability* — the server is the best
/// witness, and a hand-written value that survives a model swap silently
/// misroutes work (`urn:llm:select` ROUTES on them). `cost` and `vendor` are
/// *governance* — never discovered, only declared; see [`listing_caps`].
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
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
    /// The service behind the endpoint (`ollama`, `openai`, `anthropic`, `mlx`, …)
    /// — the governance axis: `vendor!=openai` in a `needs=` expression excludes
    /// it, and a provider that *didn't declare* a vendor can't pass the exclusion
    /// (it might be that vendor).
    #[serde(default)]
    pub vendor: Option<String>,
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
    /// Model used when a request doesn't name one — or **`None`**, meaning this
    /// provider names the **server**, not the model: the model is discovered
    /// from the backend on each resolve (see [`OpenAiConfig::discovering`]).
    pub default_model: Option<String>,
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
            default_model: Some(default_model.into()),
            api_key: None,
            caps: Caps {
                cost: Some("local".to_string()),
                vendor: Some("ollama".to_string()),
                ..Caps::default()
            },
        }
    }

    /// A provider that names the **server**, not the model: whatever that server
    /// is serving right now answers, and a model swap behind it needs no config
    /// edit and no host restart. `caps` still carries the **declared** governance
    /// (`cost`, `vendor`) — those are never taken from the server's self-report.
    pub fn discovering(provider: impl Into<String>, base_url: impl Into<String>) -> Self {
        OpenAiConfig {
            provider: provider.into(),
            base_url: base_url.into(),
            default_model: None,
            api_key: None,
            caps: Caps::default(),
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
    ///     "server": { "base_url": "http://localhost:8000/v1",
    ///                 "caps": { "cost": "local", "vendor": "mlx" } },
    ///     "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "…" }
    ///   } }
    /// ```
    ///
    /// **`model` is optional.** An entry that omits it names the *server*: the
    /// model is discovered from the backend per resolve, so swapping the model
    /// behind a server needs no config edit and no restart (the registry is read
    /// once at kernel construction — there is no watcher). Declared `caps` still
    /// win over anything the server says about itself.
    pub fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Entry {
            base_url: String,
            #[serde(default, alias = "default_model")]
            model: Option<String>,
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

/// An annotation overrode a declared value — reported so the host can log the
/// conflict (annotations are authoritative, but never silently).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnotationConflict {
    /// The provider whose trait was overridden.
    pub provider: String,
    /// The trait, e.g. `vendor`, `context`.
    pub trait_name: String,
    /// What the config declared.
    pub declared: String,
    /// What the annotation asserts (now in effect).
    pub annotated: String,
}

impl Registry {
    /// Apply annotation facts — triples from an alignment/annotation graph that
    /// complete or correct under-specified provider descriptions.
    ///
    /// Each fact is `(subject, predicate, object)`: the subject is a backend IRI
    /// (`urn:llm:<name>:ask`, as emitted by `urn:llm:models`) or a bare provider
    /// name; the predicate is an `ik:` trait (full IRI, `ik:`-prefixed, or bare —
    /// `vendor` · `cost` · `context` · `tools` · `jsonMode`/`json` · `modality` ·
    /// `params`). **Annotations are authoritative**: they fill gaps AND override
    /// declared values — every override is returned as an [`AnnotationConflict`]
    /// for the host to log. `modality` facts union in (never a conflict); unknown
    /// subjects and predicates are ignored (an annotation graph may say more
    /// about the world than this registry knows).
    pub fn apply_annotations<S: AsRef<str>>(
        &mut self,
        facts: &[(S, S, S)],
    ) -> Vec<AnnotationConflict> {
        fn local_name(predicate: &str) -> &str {
            let p = predicate
                .rsplit_once('#')
                .map_or(predicate, |(_, local)| local);
            p.rsplit_once(':').map_or(p, |(_, local)| local)
        }
        let mut conflicts = Vec::new();
        for (subject, predicate, object) in facts {
            let (subject, object) = (subject.as_ref(), object.as_ref().to_string());
            let Some(provider) = self.providers.iter_mut().find(|p| {
                subject == p.provider || subject == format!("urn:llm:{}:ask", p.provider)
            }) else {
                continue;
            };
            let mut set_str = |field: &mut Option<String>, name: &str, value: String| {
                if let Some(old) = field.as_deref() {
                    if old != value {
                        conflicts.push(AnnotationConflict {
                            provider: provider.provider.clone(),
                            trait_name: name.to_string(),
                            declared: old.to_string(),
                            annotated: value.clone(),
                        });
                    }
                }
                *field = Some(value);
            };
            match local_name(predicate.as_ref()) {
                "vendor" => set_str(&mut provider.caps.vendor, "vendor", object),
                "cost" => set_str(&mut provider.caps.cost, "cost", object),
                "params" => set_str(&mut provider.caps.params, "params", object),
                "context" => {
                    if let Ok(n) = object.parse::<u64>() {
                        if let Some(old) = provider.caps.context {
                            if old != n {
                                conflicts.push(AnnotationConflict {
                                    provider: provider.provider.clone(),
                                    trait_name: "context".to_string(),
                                    declared: old.to_string(),
                                    annotated: object,
                                });
                            }
                        }
                        provider.caps.context = Some(n);
                    }
                }
                "tools" | "jsonMode" | "json" => {
                    if let Ok(b) = object.parse::<bool>() {
                        let field = if local_name(predicate.as_ref()) == "tools" {
                            &mut provider.caps.tools
                        } else {
                            &mut provider.caps.json
                        };
                        let name = if local_name(predicate.as_ref()) == "tools" {
                            "tools"
                        } else {
                            "json"
                        };
                        if let Some(old) = *field {
                            if old != b {
                                conflicts.push(AnnotationConflict {
                                    provider: provider.provider.clone(),
                                    trait_name: name.to_string(),
                                    declared: old.to_string(),
                                    annotated: object,
                                });
                            }
                        }
                        *field = Some(b);
                    }
                }
                // Additive: modalities are a set, an assertion joins it.
                "modality" if !provider.caps.modalities.contains(&object) => {
                    provider.caps.modalities.push(object);
                }
                _ => {}
            }
        }
        conflicts
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
/// Binds `urn:llm:ask` (the facade), a `urn:llm:<provider>:ask` / `:up` /
/// `:installed` / `:model` for every provider in the registry (each directly
/// addressable and catalog-advertised with its own model/base_url), and
/// `urn:llm:config` (the effective registry, keys redacted).
/// Accepts a [`Registry`] or — via `From<OpenAiConfig>` — a single [`OpenAiConfig`].
/// The host supplies the [`HttpTransport`].
pub fn space(transport: Arc<dyn HttpTransport>, registry: impl Into<Registry>) -> EndpointSpace {
    let registry = registry.into();
    let mut space = EndpointSpace::new()
        .bind(
            Exact::new("urn:llm:ask"),
            AskFacade::new(registry.clone(), Arc::clone(&transport)),
        )
        .bind(
            Exact::new("urn:llm:config"),
            ConfigEndpoint::new(registry.clone()),
        )
        .bind(
            Exact::new("urn:llm:models"),
            ModelsEndpoint::new(registry.clone(), Arc::clone(&transport)),
        )
        .bind(
            Exact::new("urn:llm:select"),
            SelectEndpoint::new(registry.clone(), Arc::clone(&transport)),
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
            )
            .bind(
                Exact::new(format!("urn:llm:{}:installed", provider.provider)),
                InstalledEndpoint::new(provider.clone(), Arc::clone(&transport)),
            )
            .bind(
                Exact::new(format!("urn:llm:{}:model", provider.provider)),
                ModelEndpoint::new(provider.clone(), Arc::clone(&transport)),
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
    transport: Arc<dyn HttpTransport>,
}

impl AskFacade {
    /// A facade over `registry`: routes by `provider=`, else resolves `needs=`
    /// against the trait profiles (declared ⊕ discovered), else falls back to the
    /// registry's default. The transport is used only for trait discovery.
    pub fn new(registry: Registry, transport: Arc<dyn HttpTransport>) -> Self {
        AskFacade {
            registry,
            transport,
        }
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
            let effective = effective_registry(&self.registry, &self.transport, inv).await;
            select(&effective, &parse_needs(needs)?)
                .ok_or_else(|| no_match_error("urn:llm:ask", needs, &effective))?
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
            "llm-ask",
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
        // `model=` wins verbatim — always, for a discovering provider too: the
        // caller named a model, and naming one is never a request to go looking.
        // Otherwise: the configured id, else whatever the server serves now.
        let model = match inv.inline_str("model") {
            Ok(named) => named.to_string(),
            Err(_) => {
                resolve_model(
                    self.transport.as_ref(),
                    inv,
                    &self.config,
                    &format!("urn:llm:{}:ask", self.config.provider),
                )
                .await?
                .0
            }
        };

        // Label this span with what the selection actually chose — the trace
        // answers "on which model, at which provider/tier" by name instead of
        // leaving it implicit in the target IRI (a free no-op untraced).
        inv.trace_note("model", &model);
        inv.trace_note("provider", &self.config.provider);
        if let Some(cost) = &self.config.caps.cost {
            inv.trace_note("cost", cost);
        }

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

        let explicit_model = inv.inline_str("model").is_ok();
        let mut response = post_json(self.transport.as_ref(), &url, &headers, &payload).await?;

        // The configured DEFAULT model may not exist on this machine's server
        // (the demo moved hosts, the model was never pulled). A request that
        // names `model=` explicitly errors honestly — never substitute — but a
        // defaulted one resolves against what IS installed and retries once.
        if response.status == 404 && !explicit_model {
            // Smallest CHAT-capable model — an embedder may sort first (they're
            // tiny) but can't answer a chat request.
            let fallback = installed_models(self.transport.as_ref(), inv, &self.config)
                .await
                .ok()
                .and_then(|models| {
                    models
                        .into_iter()
                        .find(|m| model_supports(m, "completion"))
                        .map(|m| m.model)
                });
            if let Some(first) = fallback {
                if first != model {
                    payload["model"] = json!(first);
                    response = post_json(self.transport.as_ref(), &url, &headers, &payload).await?;
                }
            }
        }

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
            &format!("llm-{}-ask", self.config.provider),
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
                    // `null` for a provider that names the server: nothing is
                    // configured. This resource reports the REGISTRY, so it
                    // stays a pure config read — it never probes.
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
        Description::new("llm-config")
            .summary(
                "The effective LLM provider registry (default + configured backends; \
                 API keys redacted).",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .output("application/json")
    }
}

// ---- the model-identity resource --------------------------------------------

/// `urn:llm:<provider>:model` — the provider's configured model id, verbatim,
/// as `text/plain`. The cheap identity face: one small resolve answers "which
/// model actually serves this provider?" for consumers that fold TRUE model
/// identity into derived artifacts (archive version tags, provenance labels)
/// without coupling to the whole `urn:llm:config` registry JSON. Model ids are
/// not secrets (unlike the api keys `:config` redacts) — nothing is hidden.
///
/// **Two cost contracts, chosen by the provider's own config** — the resource is
/// the same, the provider decides what it costs:
///
/// * A provider that **pinned a model** answers from config: no network, no
///   capability, permanently cacheable, on the same grounds as `urn:llm:config`
///   and `urn:llm:models` (the host loads the registry at start-up, so a config
///   edit needs a restart, which also empties the cache — live reload does not
///   exist yet; if it lands, `:config`, `:models`, and `:model` all need the
///   same golden-thread cut).
/// * A provider that **names only the server** has no configured id to report,
///   so the discovered id is the only honest answer: it probes the backend
///   (`urn:cap:net:*`, declared) and is **uncacheable** — caching it would
///   reintroduce exactly the staleness discovery removes, and a cached
///   representation cannot be invalidated across a mount.
///
/// For [`ikigai-browse`]'s explain archive — which keys version tags on this
/// resource — that is the semantics it already asks for: the four pinned
/// providers keep their exact id (no archive re-derivation), and a discovering
/// provider re-keys when the model behind the server actually changes, which is
/// the stated feature ("a model swap re-keys the archive without any
/// browse-side config").
///
/// [`ikigai-browse`]: https://github.com/ikigai-rs/ikigai-browse
pub struct ModelEndpoint {
    config: OpenAiConfig,
    transport: Arc<dyn HttpTransport>,
}

impl ModelEndpoint {
    fn new(config: OpenAiConfig, transport: Arc<dyn HttpTransport>) -> Self {
        ModelEndpoint { config, transport }
    }
}

#[async_trait]
impl Endpoint for ModelEndpoint {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        fn plain(model: String) -> Representation {
            Representation::new(
                ReprType::new("text/plain").with_param("charset", "utf-8"),
                model.into_bytes(),
            )
        }
        match &self.config.default_model {
            Some(model) => Ok(plain(model.clone()).cacheable()),
            None => {
                let (model, _) = resolve_model(
                    self.transport.as_ref(),
                    inv,
                    &self.config,
                    &format!("urn:llm:{}:model", self.config.provider),
                )
                .await?;
                Ok(plain(model)) // live fact: uncacheable
            }
        }
    }

    fn name(&self) -> &str {
        "llm-model"
    }

    fn describe(&self) -> Description {
        let description = Description::new(format!("llm-{}-model", self.config.provider))
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .output("text/plain;charset=utf-8");
        // Declared = enforced: only the discovering form touches the network, so
        // only it declares the net capability (declaring it on the config read
        // would make the manifold over-offer).
        if self.config.default_model.is_some() {
            description.summary(
                "The provider's configured model id, verbatim (text/plain) — the cheap \
                 identity face for consumers folding true model identity into derived \
                 artifacts (archive version tags, provenance labels) without pulling \
                 the whole urn:llm:config registry. No network; cacheable.",
            )
        } else {
            description
                .summary(
                    "The model this provider's SERVER is serving right now (text/plain) \
                     — this provider pinned no model, so the discovered id is the only \
                     honest identity. Probes the backend; uncacheable.",
                )
                .requires(CAP_NET)
        }
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
    transport: Arc<dyn HttpTransport>,
}

impl ModelsEndpoint {
    fn new(registry: Registry, transport: Arc<dyn HttpTransport>) -> Self {
        ModelsEndpoint {
            registry,
            transport,
        }
    }

    fn as_json(registry: &Registry) -> Value {
        let mut models = serde_json::Map::new();
        for p in &registry.providers {
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
                        "vendor": caps.vendor,
                        "params": caps.params,
                    },
                }),
            );
        }
        json!({ "default": registry.default, "models": models })
    }

    /// The trait graph. Vocabulary is module-local for now (`ik:LlmBackend`,
    /// `ik:model`, `ik:context`, `ik:modality`, `ik:tools`, `ik:jsonMode`,
    /// `ik:cost`, `ik:params`) — promotion into ikigai-vocab is a follow-up.
    fn as_turtle(registry: &Registry) -> String {
        let mut ttl = String::from("@prefix ik: <https://ikigai-rs.dev/ns#> .\n");
        for p in &registry.providers {
            let mut props = vec!["a ik:LlmBackend".to_string()];
            // Absent when a discovering provider's backend could not be reached:
            // the inventory still lists it, rather than failing the whole graph.
            if let Some(model) = &p.default_model {
                props.push(format!("ik:model {}", ttl_str(model)));
            }
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
            if let Some(v) = &p.caps.vendor {
                props.push(format!("ik:vendor {}", ttl_str(v)));
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
            registry.default
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
        // The inventory is declared ⊕ discovered — gaps filled live from the
        // provider (Ollama /api/show) where declared vendor + capability allow.
        let effective = effective_registry(&self.registry, &self.transport, inv).await;
        let want_turtle = inv
            .inline_str("as")
            .map(|s| s.contains("turtle"))
            .unwrap_or(false);
        let (media, bytes) = if want_turtle {
            ("text/turtle", Self::as_turtle(&effective).into_bytes())
        } else {
            (
                "application/json",
                serde_json::to_vec(&Self::as_json(&effective)).unwrap_or_default(),
            )
        };
        Ok(with_cacheability(
            Representation::new(ReprType::new(media).with_param("charset", "utf-8"), bytes),
            &self.registry,
        ))
    }

    fn name(&self) -> &str {
        "llm-models"
    }

    fn describe(&self) -> Description {
        Description::new("llm-models")
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

// ---- trait discovery (Ollama /api/show) --------------------------------------

/// Traits discovered from Ollama's native `/api/show`: context length (from
/// `model_info.*.context_length`), modalities and tool support (from
/// `capabilities`), and the parameter size. Only attempted for providers that
/// **declare** `vendor: "ollama"` — that declaration is the opt-in that the
/// native API exists; we never probe an unknown vendor with a model name. Only
/// where the capability allows the host; graceful on any failure (None).
async fn discovered_caps(
    transport: &dyn HttpTransport,
    inv: &Invocation<'_>,
    provider: &OpenAiConfig,
) -> Option<Caps> {
    if provider.caps.vendor.as_deref() != Some("ollama") {
        return None;
    }
    // The native API lives at the server root, not under the OpenAI-compat /v1.
    let root = provider
        .base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1");
    let url = format!("{root}/api/show");
    let parsed = url::Url::parse(&url).ok()?;
    let host = parsed.host_str().unwrap_or("");
    if !ikigai_http::net_allows(inv.capability, host, parsed.path()) {
        return None; // no capability -> no discovery; declared profile stands
    }
    // Needs a model to ask ABOUT. `effective_registry` resolves a discovering
    // provider's model first, so by here it is known — unless the backend was
    // unreachable, in which case there is nothing to show.
    let model = provider.default_model.as_deref()?;
    let body = serde_json::to_vec(&json!({ "model": model })).ok()?;
    let response = transport
        .send(HttpRequest {
            method: Method::Post,
            url,
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body,
        })
        .await
        .ok()?;
    if response.status >= 400 {
        return None;
    }
    let v: Value = serde_json::from_slice(&response.body).ok()?;
    let mut caps = Caps::default();
    if let Some(info) = v["model_info"].as_object() {
        caps.context = info
            .iter()
            .find(|(k, _)| k.ends_with(".context_length"))
            .and_then(|(_, val)| val.as_u64());
    }
    if let Some(list) = v["capabilities"].as_array() {
        let has = |name: &str| list.iter().any(|c| c.as_str() == Some(name));
        caps.modalities = if has("vision") {
            vec!["text".to_string(), "vision".to_string()]
        } else {
            vec!["text".to_string()]
        };
        caps.tools = Some(has("tools"));
    }
    caps.params = v["details"]["parameter_size"]
        .as_str()
        .map(|s| s.to_string());
    Some(caps)
}

/// Declared ⊕ discovered, **declared-wins**: discovery only fills gaps. (The
/// annotation layer sits ABOVE declared and may override — the precedence
/// ladder is annotations > declared > discovered.)
fn merge_declared_wins(declared: &Caps, discovered: Caps) -> Caps {
    Caps {
        context: declared.context.or(discovered.context),
        modalities: if declared.modalities.is_empty() {
            discovered.modalities
        } else {
            declared.modalities.clone()
        },
        tools: declared.tools.or(discovered.tools),
        json: declared.json.or(discovered.json),
        cost: declared.cost.clone().or(discovered.cost),
        vendor: declared.vendor.clone().or(discovered.vendor),
        params: declared.params.clone().or(discovered.params),
    }
}

/// This invocation's view of the registry: every provider's declared profile
/// with discovered gaps filled. What `urn:llm:models` reports and selection
/// reasons over.
async fn effective_registry(
    registry: &Registry,
    transport: &Arc<dyn HttpTransport>,
    inv: &Invocation<'_>,
) -> Registry {
    let mut effective = registry.clone();
    for provider in &mut effective.providers {
        // A provider that named only the SERVER: ask the server what it serves.
        // The same round trip carries that model's advertised capability facts,
        // so discovery costs one request, not two. Deliberately GRACEFUL here —
        // an unreachable backend leaves this entry's model unresolved and the
        // inventory still describes every other provider. `:ask` and `:model`
        // are where an unreachable backend fails loudly, and they must.
        if provider.default_model.is_none() {
            if let Ok((model, advertised)) =
                resolve_model(transport.as_ref(), inv, provider, "urn:llm:models").await
            {
                provider.default_model = Some(model);
                if let Some(found) = advertised {
                    provider.caps = merge_declared_wins(&provider.caps, found);
                }
            }
        }
        if let Some(found) = discovered_caps(transport.as_ref(), inv, provider).await {
            provider.caps = merge_declared_wins(&provider.caps, found);
        }
    }
    effective
}

/// Does any provider name only its server? Then this invocation had to ask the
/// network what the answer is, and the result is a **live fact** — cacheing it
/// would restore exactly the staleness discovery exists to remove (and a cached
/// representation reached through a mount can never be invalidated at all).
/// A registry of pinned providers is unaffected: config in, cacheable out.
fn any_discovering(registry: &Registry) -> bool {
    registry.providers.iter().any(|p| p.default_model.is_none())
}

/// Mark a config-derived representation cacheable — unless discovery fed it.
fn with_cacheability(repr: Representation, registry: &Registry) -> Representation {
    if any_discovering(registry) {
        repr
    } else {
        repr.cacheable()
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
    /// `vendor=ollama` — the declared vendor must be exactly this.
    VendorIs(String),
    /// `vendor!=openai` — governance exclusion: the declared vendor must differ.
    /// A provider that declared NO vendor fails this (it might be that vendor).
    VendorNot(String),
    /// `provider=fast` — this registry entry by name.
    ProviderIs(String),
    /// `provider!=posh` — any registry entry but this one.
    ProviderNot(String),
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
        } else if let Some((k, v)) = term.split_once("!=") {
            match k.trim() {
                "vendor" => Need::VendorNot(v.trim().to_string()),
                "provider" => Need::ProviderNot(v.trim().to_string()),
                other => return Err(bad(term, &format!("`{other}` does not support !="))),
            }
        } else if let Some((k, v)) = term.split_once('=') {
            match k.trim() {
                "cost" => Need::CostExactly(
                    cost_rank(v.trim()).ok_or_else(|| bad(term, "expected local|cheap|premium"))?,
                ),
                "modality" => Need::Modality(v.trim().to_string()),
                "vendor" => Need::VendorIs(v.trim().to_string()),
                "provider" => Need::ProviderIs(v.trim().to_string()),
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

/// Does a provider satisfy every need? Conservative: a trait it didn't declare
/// cannot satisfy a requirement on it — including a `vendor!=` exclusion, which
/// an undeclared vendor FAILS (it might be the excluded vendor).
fn satisfies(provider: &OpenAiConfig, needs: &[Need]) -> bool {
    let caps = &provider.caps;
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
        Need::VendorIs(v) => caps.vendor.as_deref() == Some(v.as_str()),
        Need::VendorNot(v) => caps.vendor.as_deref().is_some_and(|x| x != v),
        Need::ProviderIs(n) => provider.provider == *n,
        Need::ProviderNot(n) => provider.provider != *n,
    })
}

/// The satisfying provider under the policy **cheapest-that-fits → smallest
/// context → registry order** (an undeclared cost tier sorts last).
fn select<'a>(registry: &'a Registry, needs: &[Need]) -> Option<&'a OpenAiConfig> {
    registry
        .providers
        .iter()
        .enumerate()
        .filter(|(_, p)| satisfies(p, needs))
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
    transport: Arc<dyn HttpTransport>,
}

impl SelectEndpoint {
    fn new(registry: Registry, transport: Arc<dyn HttpTransport>) -> Self {
        SelectEndpoint {
            registry,
            transport,
        }
    }
}

#[async_trait]
impl Endpoint for SelectEndpoint {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let needs = inv.inline_str("needs")?;
        let effective = effective_registry(&self.registry, &self.transport, inv).await;
        let winner = select(&effective, &parse_needs(needs)?)
            .ok_or_else(|| no_match_error("urn:llm:select", needs, &effective))?;
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
                "vendor": winner.caps.vendor,
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
        Ok(with_cacheability(
            Representation::new(ReprType::new(media).with_param("charset", "utf-8"), bytes),
            &self.registry,
        ))
    }

    fn name(&self) -> &str {
        "llm-select"
    }

    fn describe(&self) -> Description {
        Description::new("llm-select")
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
                 modality=X (or bare text/vision/audio) · tools · json · vendor=X · \
                 vendor!=X (governance: e.g. no openai) · provider=name · provider!=name",
            ))
            .input(
                ArgSpec::new("as")
                    .summary("text/plain backend IRI (default) or application/json detail")
                    .optional(),
            )
            .output("text/plain;charset=utf-8")
    }
}

/// POST a JSON payload to a provider, mapping transport failure to an endpoint
/// error (a 4xx/5xx is still a response — callers decide).
async fn post_json(
    transport: &dyn HttpTransport,
    url: &str,
    headers: &[(String, String)],
    payload: &Value,
) -> Result<HttpResponse> {
    let body = serde_json::to_vec(payload)
        .map_err(|e| Error::Endpoint(format!("llm: encoding request failed: {e}")))?;
    transport
        .send(HttpRequest {
            method: Method::Post,
            url: url.to_string(),
            headers: headers.to_vec(),
            body,
        })
        .await
        .map_err(|e| Error::Endpoint(format!("llm transport error: {e}")))
}

/// One installed model: its name and — where the provider reports it — its
/// size in bytes, so hosts can make machine-fitness decisions (co-load budgets,
/// smallest-first defaults).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstalledModel {
    /// The model id, e.g. `llama3.2:3b`.
    pub model: String,
    /// Size in bytes, when the provider reports one (Ollama's `/api/tags` does;
    /// the OpenAI-compat listing doesn't).
    pub size: Option<u64>,
    /// What the model can do (`completion`, `embedding`, `tools`, …), where the
    /// provider reports it (Ollama's `/api/show`). Empty = unknown, and unknown
    /// passes every `supports` check — providers that don't report capabilities
    /// keep working.
    ///
    /// **Ollama's vocabulary only.** An OpenAI-compat listing may carry a
    /// `capabilities` array too, but it is a DIFFERENT vocabulary (rapid-mlx
    /// answers `["text","tools"]` — modality plus tools, no `completion` term),
    /// and folding it in here would make every such model fail
    /// `supports=completion` and vanish from the very list discovery reads.
    /// Those facts go to [`InstalledModel::advertised`] instead.
    pub capabilities: Vec<String>,
    /// Capability facts the listing advertised **about this model** — context
    /// window, modalities, tool support — where the server reports them
    /// (rapid-mlx's `/v1/models` does; Ollama's `/api/tags` and the plain
    /// OpenAI-compat listing don't). Never carries `cost` or `vendor`: those
    /// are governance, and a server is not a trustworthy witness about itself.
    pub advertised: Option<Caps>,
}

/// The capability facts an OpenAI-compat listing entry advertises about its
/// model — `context_window`, `modality`, and a `capabilities` array (rapid-mlx
/// serves all three). `None` when the entry says nothing beyond its id.
///
/// **`cost` and `vendor` are structurally absent here, and that is the point.**
/// They are governance, not capability: `vendor` is the axis a `needs=`
/// exclusion works on (`vendor!=openai`), and a provider that declared no
/// vendor cannot pass an exclusion *because it might be that vendor*. A server
/// that self-reports `owned_by: "rapid-mlx"` must not be able to launder itself
/// past a policy by saying so — so the discovered profile has no field to put
/// it in, rather than a precedence rule that could later be reordered.
fn listing_caps(entry: &Value) -> Option<Caps> {
    let mut caps = Caps::default();
    let mut found = false;
    if let Some(context) = entry["context_window"].as_u64() {
        caps.context = Some(context);
        found = true;
    }
    if let Some(modality) = entry["modality"].as_str() {
        caps.modalities.push(modality.to_string());
        found = true;
    }
    if let Some(list) = entry["capabilities"].as_array() {
        let has = |name: &str| list.iter().any(|c| c.as_str() == Some(name));
        caps.tools = Some(has("tools"));
        // The array mixes tool support with modality names; take the modalities
        // it names and leave the rest (an unknown term is not a fact we model).
        for modality in ["text", "vision", "audio"] {
            if has(modality) && !caps.modalities.iter().any(|m| m == modality) {
                caps.modalities.push(modality.to_string());
            }
        }
        found = true;
    }
    found.then_some(caps)
}

/// The model a provider serves, and what the server says about it: the
/// **configured** id when the provider pinned one (no network — a pinned
/// provider costs exactly what it did before), else the model the backend is
/// serving **right now**.
///
/// Discovery reuses the one ordering rule the crate already has —
/// [`installed_models`] is smallest-first and `supports=completion`-filtered —
/// so a big model stays an explicit choice (`model=` / `needs=`) and never an
/// accident of list order. More than one model served is a legitimate state,
/// not an error: the ordering rule resolves it (rapid-mlx lists its canonical
/// id and a lowercase alias for the same weights today, so erroring would break
/// a live server for no gain). Pin a `model` to say which one you meant.
///
/// Failure names the `base_url` and stops. There is deliberately **no
/// cross-provider fallback**: silently answering from a different backend would
/// defeat every governance term a caller expressed.
async fn resolve_model(
    transport: &dyn HttpTransport,
    inv: &Invocation<'_>,
    config: &OpenAiConfig,
    who: &str,
) -> Result<(String, Option<Caps>)> {
    if let Some(model) = &config.default_model {
        return Ok((model.clone(), None));
    }
    let models = installed_models(transport, inv, config)
        .await
        .map_err(|e| {
            Error::Endpoint(format!(
                "{who}: could not discover a model at `{}` — this provider declares no \
                 `model`, and listing the server's models failed: {e}",
                config.base_url
            ))
        })?;
    models
        .into_iter()
        .find(|m| model_supports(m, "completion"))
        .map(|m| (m.model, m.advertised))
        .ok_or_else(|| {
            Error::Endpoint(format!(
                "{who}: could not discover a model at `{}` — this provider declares no \
                 `model`, and the server listed no chat-capable model",
                config.base_url
            ))
        })
}

/// Whether a model can serve `what` (`completion`, `embedding`, …). Unknown
/// capabilities pass: no facts, no policy.
fn model_supports(m: &InstalledModel, what: &str) -> bool {
    m.capabilities.is_empty() || m.capabilities.iter().any(|c| c == what)
}

/// The models actually available at the provider right now, **smallest-first**
/// where sizes are known. A declared `vendor: "ollama"` uses the native
/// `/api/tags` (which reports sizes); anything else — or a tags failure — falls
/// back to the OpenAI-compat `GET {base}/models` listing (names only, server
/// order). Net-gated like every provider call. Backs
/// `urn:llm:<provider>:installed` and the default-model fallback — so "first
/// installed" means "cheapest to run": big models are an explicit choice
/// (`model=` / `needs=`), never an accident of list order.
async fn installed_models(
    transport: &dyn HttpTransport,
    inv: &Invocation<'_>,
    config: &OpenAiConfig,
) -> Result<Vec<InstalledModel>> {
    if config.caps.vendor.as_deref() == Some("ollama") {
        if let Ok(mut models) = installed_via_tags(transport, inv, config).await {
            if !models.is_empty() {
                models.sort_by_key(|m| m.size.unwrap_or(u64::MAX));
                annotate_capabilities(transport, inv, config, &mut models).await;
                return Ok(models);
            }
        }
    }
    let url = require_net(
        inv,
        &config.base_url,
        "models",
        &format!("urn:llm:{}:installed", config.provider),
    )?;
    let response = get(transport, url).await?;
    if response.status >= 400 {
        return Err(Error::Endpoint(format!(
            "urn:llm:{}:installed: provider returned {}",
            config.provider, response.status
        )));
    }
    let v: Value = serde_json::from_slice(&response.body)
        .map_err(|e| Error::Endpoint(format!("llm: model list was not JSON: {e}")))?;
    Ok(v["data"]
        .as_array()
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m["id"].as_str().map(|id| (id, m)))
                .map(|(id, entry)| InstalledModel {
                    model: id.to_string(),
                    size: None,
                    capabilities: Vec::new(),
                    advertised: listing_caps(entry),
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Enrich an Ollama listing with per-model capabilities via `/api/show` — the
/// fact that separates chat models from embedders. Best-effort: any failure
/// leaves that model's capabilities unknown (empty), which every check treats
/// as capable.
async fn annotate_capabilities(
    transport: &dyn HttpTransport,
    inv: &Invocation<'_>,
    config: &OpenAiConfig,
    models: &mut [InstalledModel],
) {
    let root = config
        .base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1");
    let url = format!("{root}/api/show");
    let Ok(parsed) = url::Url::parse(&url) else {
        return;
    };
    let host = parsed.host_str().unwrap_or("");
    if !ikigai_http::net_allows(inv.capability, host, parsed.path()) {
        return;
    }
    let headers = [("Content-Type".to_string(), "application/json".to_string())];
    for m in models.iter_mut() {
        let Ok(response) = post_json(transport, &url, &headers, &json!({ "model": m.model })).await
        else {
            continue;
        };
        if response.status >= 400 {
            continue;
        }
        let Ok(v) = serde_json::from_slice::<Value>(&response.body) else {
            continue;
        };
        if let Some(caps) = v["capabilities"].as_array() {
            m.capabilities = caps
                .iter()
                .filter_map(|c| c.as_str().map(str::to_string))
                .collect();
        }
    }
}

/// Ollama's native listing — names AND sizes. Only called for a declared
/// ollama vendor (the same opt-in rule as trait discovery).
async fn installed_via_tags(
    transport: &dyn HttpTransport,
    inv: &Invocation<'_>,
    config: &OpenAiConfig,
) -> Result<Vec<InstalledModel>> {
    let root = config
        .base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1");
    let url = format!("{root}/api/tags");
    let parsed = url::Url::parse(&url)
        .map_err(|e| Error::Endpoint(format!("llm: bad base_url `{}`: {e}", config.base_url)))?;
    let host = parsed.host_str().unwrap_or("");
    if !ikigai_http::net_allows(inv.capability, host, parsed.path()) {
        return Err(Error::Endpoint(format!(
            "urn:llm:{}:installed: capability does not allow reaching `{host}`",
            config.provider
        )));
    }
    let response = get(transport, url).await?;
    if response.status >= 400 {
        return Err(Error::Endpoint(format!(
            "llm: /api/tags returned {}",
            response.status
        )));
    }
    let v: Value = serde_json::from_slice(&response.body)
        .map_err(|e| Error::Endpoint(format!("llm: /api/tags was not JSON: {e}")))?;
    Ok(v["models"]
        .as_array()
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    m["name"].as_str().map(|name| InstalledModel {
                        model: name.to_string(),
                        size: m["size"].as_u64(),
                        capabilities: Vec::new(),
                        // Ollama's tags listing carries no capability facts;
                        // `/api/show` fills them via `discovered_caps`.
                        advertised: None,
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

/// A capability-checked-elsewhere GET, mapping transport failure to an error.
async fn get(transport: &dyn HttpTransport, url: String) -> Result<HttpResponse> {
    transport
        .send(HttpRequest {
            method: Method::Get,
            url,
            headers: Vec::new(),
            body: Vec::new(),
        })
        .await
        .map_err(|e| Error::Endpoint(format!("llm transport error: {e}")))
}

/// `urn:llm:<provider>:installed` — what the provider can actually serve right
/// now, as a newline list (pipeable; `as=application/json` for an array). The
/// complement of `urn:llm:models`: models/config say what's CONFIGURED, this
/// says what's INSTALLED. Uncacheable — a live fact (pulls change it).
pub struct InstalledEndpoint {
    config: OpenAiConfig,
    transport: Arc<dyn HttpTransport>,
}

impl InstalledEndpoint {
    fn new(config: OpenAiConfig, transport: Arc<dyn HttpTransport>) -> Self {
        InstalledEndpoint { config, transport }
    }
}

#[async_trait]
impl Endpoint for InstalledEndpoint {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let mut models = installed_models(self.transport.as_ref(), inv, &self.config).await?;
        if let Ok(want) = inv.inline_str("supports") {
            models.retain(|m| model_supports(m, want));
        }
        let want_json = inv
            .inline_str("as")
            .map(|s| s.contains("json"))
            .unwrap_or(false);
        let (media, bytes) = if want_json {
            let detail: Vec<Value> = models
                .iter()
                .map(
                    |m| json!({ "model": m.model, "size": m.size, "capabilities": m.capabilities }),
                )
                .collect();
            (
                "application/json",
                serde_json::to_vec(&detail).unwrap_or_default(),
            )
        } else {
            let names: Vec<&str> = models.iter().map(|m| m.model.as_str()).collect();
            ("text/plain", names.join("\n").into_bytes())
        };
        Ok(Representation::new(
            ReprType::new(media).with_param("charset", "utf-8"),
            bytes,
        ))
    }

    fn name(&self) -> &str {
        "llm-installed"
    }

    fn describe(&self) -> Description {
        Description::new(format!("llm-{}-installed", self.config.provider))
            .summary(
                "The models the provider can actually serve right now (newline list; \
                 as=application/json for an array). Live fact — uncacheable. The \
                 complement of urn:llm:models: configured vs installed.",
            )
            .verb(Verb::Source)
            .verb(Verb::Meta)
            .input(
                ArgSpec::new("as")
                    .summary("text/plain newline list (default) or application/json")
                    .optional(),
            )
            .input(
                ArgSpec::new("supports")
                    .summary(
                        "keep only models with this capability (completion, embedding, \
                         tools, …); models whose capabilities are unknown always pass",
                    )
                    .optional(),
            )
            .output("text/plain;charset=utf-8")
            .requires(CAP_NET)
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
        Description::new(format!("llm-{}-up", self.config.provider))
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
    fn description_ids_are_labels_not_iris() {
        // A `Description::id` is the endpoint's LABEL — it names the
        // implementation, while the binding IRI names the resource. Using the
        // IRI for both made a wire listing read `urn:llm:ask → urn:llm:ask`
        // (scoped_entries pairs endpoint-IRI with id) and produced malformed
        // catalog skolems (`urn:ikigai:endpoint:urn:llm:ask:action:source`).
        // The id must match `name()`, as everywhere else in the ecosystem.
        let mock = MockTransport::new(CANNED);
        let config = OpenAiConfig::ollama("llama3.1");
        let facade = AskFacade::new(Registry::single(config.clone()), Arc::clone(&mock) as _);
        let backend = OpenAiBackend::new(config.clone(), Arc::clone(&mock) as _);
        let installed = InstalledEndpoint::new(config.clone(), Arc::clone(&mock) as _);
        let model = ModelEndpoint::new(config.clone(), Arc::clone(&mock) as _);
        let up = UpEndpoint::new(config, Arc::clone(&mock) as _);

        for (id, name) in [
            (facade.describe().id, facade.name().to_string()),
            (backend.describe().id, backend.name().to_string()),
            (installed.describe().id, installed.name().to_string()),
            (model.describe().id, model.name().to_string()),
            (up.describe().id, up.name().to_string()),
        ] {
            assert!(
                !id.starts_with("urn:"),
                "a description id is a label, not an IRI: {id}"
            );
            let _ = name; // name()s are per-KIND; ids are per-BINDING (provider-qualified)
        }
        assert_eq!(facade.describe().id, "llm-ask");
        assert_eq!(backend.describe().id, "llm-ollama-ask");
        assert_eq!(installed.describe().id, "llm-ollama-installed");
        assert_eq!(model.describe().id, "llm-ollama-model");
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
    fn a_traced_ask_labels_model_provider_and_cost() {
        struct Rec(Mutex<Vec<ikigai_core::TraceEvent>>);
        impl ikigai_core::Tracer for Rec {
            fn record(&self, event: ikigai_core::TraceEvent) {
                self.0.lock().unwrap().push(event);
            }
        }
        let mock = MockTransport::new(CANNED);
        let kernel = kernel_with(Arc::clone(&mock));
        let rec = Arc::new(Rec(Mutex::new(Vec::new())));
        block_on(kernel.issue_traced(
            ask("urn:llm:ollama:ask", "hi"),
            &Capability::root(),
            rec.clone(),
        ))
        .unwrap();
        let events = rec.0.lock().unwrap().clone();
        let ask_event = events
            .iter()
            .find(|e| e.target == "urn:llm:ollama:ask")
            .expect("the ask span is recorded");
        // The trace answers "on which model, at which provider/tier" by name.
        let has = |k: &str, v: &str| ask_event.notes.contains(&(k.to_string(), v.to_string()));
        assert!(has("model", "llama3.1"), "notes: {:?}", ask_event.notes);
        assert!(has("provider", "ollama"));
        assert!(has("cost", "local"));
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
        assert_eq!(remote.default_model.as_deref(), Some("gpt-4o"));
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
            "urn:llm:fast:model",
            "urn:llm:remote:model",
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
                      "caps": { "context": 131072, "modalities": ["text"], "cost": "local", "vendor": "ollama" } },
            "seer": { "base_url": "http://localhost:11434/v1", "model": "llava:7b",
                      "caps": { "context": 32768, "modalities": ["text", "vision"], "cost": "local", "vendor": "ollama" } },
            "posh": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "sk-S",
                      "caps": { "context": 131072, "modalities": ["text", "vision"], "tools": true, "cost": "premium", "vendor": "openai" } }
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
    fn vendor_and_provider_constraints_apply() {
        let kernel = select_kernel(MockTransport::new(CANNED));
        // Governance: no openai. posh is excluded; among fast/seer (both local)
        // the smaller context wins.
        assert_eq!(selected(&kernel, "vendor!=openai"), "urn:llm:seer:ask");
        // ...and combined with a trait only posh has -> nothing fits.
        let none = block_on(
            kernel.issue(
                Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                    .with_arg("needs", ArgRef::Inline(b"tools, vendor!=openai".to_vec())),
                &Capability::root(),
            ),
        );
        assert!(none.is_err(), "tools only exists at openai here");
        // Vendor inclusion and provider-name terms.
        assert_eq!(selected(&kernel, "vendor=openai"), "urn:llm:posh:ask");
        assert_eq!(selected(&kernel, "provider=fast"), "urn:llm:fast:ask");
        assert_eq!(
            selected(&kernel, "provider!=seer, vendor!=openai"),
            "urn:llm:fast:ask"
        );
    }

    #[test]
    fn an_undeclared_vendor_cannot_pass_a_vendor_exclusion() {
        // TWO_PROVIDERS' `remote` declares NO vendor: it must not pass
        // `vendor!=openai` — it might BE openai. (fast declares none either, so
        // nothing matches.)
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(CANNED), reg)));
        let none = block_on(
            kernel.issue(
                Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                    .with_arg("needs", ArgRef::Inline(b"vendor!=openai".to_vec())),
                &Capability::root(),
            ),
        );
        assert!(none.is_err(), "undeclared vendor must fail the exclusion");
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

    /// What Ollama's native `/api/show` answers for a 3B tool-capable model.
    const SHOW: &str = r#"{"details":{"parameter_size":"3.2B"},"model_info":{"llama.context_length":131072},"capabilities":["completion","tools"]}"#;

    #[test]
    fn discovery_fills_gaps_and_declared_wins() {
        // Declared: vendor ollama + cost local + context 999 — nothing else.
        let reg = Registry::from_json(
            r#"{ "default": "o", "providers": { "o": {
                "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
                "caps": { "vendor": "ollama", "cost": "local", "context": 999 } } } }"#,
        )
        .unwrap();
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(SHOW), reg)));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap()),
        );
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        let caps = &v["models"]["o"]["caps"];
        assert_eq!(
            caps["context"], 999,
            "declared context wins over discovered"
        );
        assert_eq!(caps["tools"], true, "tools discovered from /api/show");
        assert_eq!(caps["params"], "3.2B", "params discovered");
        assert_eq!(caps["cost"], "local", "declared cost untouched");
    }

    #[test]
    fn discovery_only_probes_declared_ollama_vendors() {
        // No vendor declared -> no /api/show probe is ever sent.
        let reg = Registry::from_json(
            r#"{ "default": "x", "providers": { "x": {
                "base_url": "http://localhost:8000/v1", "model": "m" } } }"#,
        )
        .unwrap();
        let mock = MockTransport::new(SHOW);
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, reg)));
        let _ = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap()),
        );
        assert!(
            mock.last.lock().unwrap().is_none(),
            "an unknown vendor must not be probed"
        );
    }

    #[test]
    fn discovery_failure_degrades_to_declared() {
        let kernel = Kernel::new(Arc::new(space(
            Arc::new(DownTransport),
            OpenAiConfig::ollama("llama3.2:3b"),
        )));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap()),
        );
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        assert_eq!(v["models"]["ollama"]["caps"]["cost"], "local");
    }

    #[test]
    fn selection_reasons_over_discovered_traits() {
        // `tools` is NOT declared — only discovery (/api/show) knows it.
        let reg = Registry::from_json(
            r#"{ "default": "o", "providers": { "o": {
                "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
                "caps": { "vendor": "ollama" } } } }"#,
        )
        .unwrap();
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(SHOW), reg)));
        assert_eq!(selected(&kernel, "tools"), "urn:llm:o:ask");
    }

    #[test]
    fn annotations_fill_gaps_and_override_with_conflicts() {
        let mut reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let conflicts = reg.apply_annotations(&[
            // gap-fill on `remote` (no vendor declared) — full trait-graph forms
            (
                "urn:llm:remote:ask",
                "https://ikigai-rs.dev/ns#vendor",
                "openai",
            ),
            // override on `fast` (declared 131072) — bare forms work too
            ("fast", "context", "32768"),
            // modality unions in, never conflicts
            ("fast", "ik:modality", "vision"),
            // unknown subject: ignored
            ("urn:llm:nobody:ask", "vendor", "acme"),
        ]);
        assert_eq!(
            conflicts,
            vec![AnnotationConflict {
                provider: "fast".to_string(),
                trait_name: "context".to_string(),
                declared: "131072".to_string(),
                annotated: "32768".to_string(),
            }]
        );
        let fast = reg.providers.iter().find(|p| p.provider == "fast").unwrap();
        assert_eq!(fast.caps.context, Some(32768), "annotation overrode");
        assert!(fast.caps.modalities.contains(&"vision".to_string()));
        let remote = reg
            .providers
            .iter()
            .find(|p| p.provider == "remote")
            .unwrap();
        assert_eq!(remote.caps.vendor.as_deref(), Some("openai"), "gap filled");
    }

    /// A transport that answers from a scripted queue (and logs every request)
    /// — for flows that make several calls, like the default-model fallback.
    struct QueueTransport {
        responses: Mutex<std::collections::VecDeque<HttpResponse>>,
        log: Mutex<Vec<HttpRequest>>,
    }

    impl QueueTransport {
        fn new(responses: Vec<(u16, &str)>) -> Arc<Self> {
            Arc::new(QueueTransport {
                responses: Mutex::new(
                    responses
                        .into_iter()
                        .map(|(status, body)| HttpResponse {
                            status,
                            headers: vec![],
                            body: body.as_bytes().to_vec(),
                        })
                        .collect(),
                ),
                log: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl HttpTransport for QueueTransport {
        async fn send(&self, request: HttpRequest) -> std::result::Result<HttpResponse, String> {
            self.log.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| "queue exhausted".to_string())
        }
    }

    const NOT_FOUND: &str = r#"{"error":{"message":"model "ghost" not found, try pulling it first","type":"api_error"}}"#;
    const INSTALLED: &str = r#"{"object":"list","data":[{"id":"real:latest"},{"id":"other:7b"}]}"#;
    const TAGS: &str = r#"{"models":[{"name":"big:latest","size":23900000000},{"name":"small:3b","size":2000000000}]}"#;
    const SHOW_EMBED: &str = r#"{"capabilities":["embedding"]}"#;
    const SHOW_CHAT: &str = r#"{"capabilities":["completion","tools"]}"#;

    #[test]
    fn a_missing_default_model_falls_back_to_whats_installed() {
        // The configured default ("ghost") isn't on this machine's server — the
        // demo-moved-hosts case. Expect: 404 -> list installed (native tags for
        // an ollama vendor, capabilities via /api/show smallest-first) -> retry
        // with the smallest CHAT-CAPABLE model. small:3b sorts first but is an
        // embedder — it must be passed over, not asked to chat.
        let mock = QueueTransport::new(vec![
            (404, NOT_FOUND),
            (200, TAGS),
            (200, SHOW_EMBED), // small:3b
            (200, SHOW_CHAT),  // big:latest
            (200, CANNED),
        ]);
        let mut config = OpenAiConfig::ollama("ghost");
        config.provider = "o".to_string();
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, config)));
        let out = issue(&kernel, ask("urn:llm:o:ask", "hi"));
        assert_eq!(out.bytes, b"Hello there!");
        let log = mock.log.lock().unwrap();
        assert_eq!(log.len(), 5, "chat, list, show x2, retry");
        assert!(log[1].url.ends_with("/api/tags"), "listed via native tags");
        assert!(log[2].url.ends_with("/api/show"), "capabilities probed");
        let retry = String::from_utf8_lossy(&log[4].body).to_string();
        assert!(
            retry.contains("big:latest"),
            "retried with the smallest chat-capable model, not the embedder: {retry}"
        );
    }

    #[test]
    fn supports_filters_out_models_that_cannot_chat() {
        // The selection seam the jury demo rides: supports=completion drops the
        // embedder even though it sorts first (smallest).
        let mock = QueueTransport::new(vec![(200, TAGS), (200, SHOW_EMBED), (200, SHOW_CHAT)]);
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, OpenAiConfig::ollama("llama3.1"))));
        let out = issue(
            &kernel,
            Request::new(
                Verb::Source,
                Iri::parse("urn:llm:ollama:installed").unwrap(),
            )
            .with_arg("supports", ArgRef::Inline(b"completion".to_vec())),
        );
        assert_eq!(out.bytes, b"big:latest");
    }

    #[test]
    fn unknown_capabilities_pass_the_supports_filter() {
        // The OpenAI-compat listing reports no capabilities: supports= must not
        // empty the list (no facts, no policy).
        let mock = MockTransport::new(INSTALLED);
        let kernel = kernel_with(mock);
        let out = issue(
            &kernel,
            Request::new(
                Verb::Source,
                Iri::parse("urn:llm:ollama:installed").unwrap(),
            )
            .with_arg("supports", ArgRef::Inline(b"completion".to_vec())),
        );
        assert_eq!(out.bytes, b"real:latest\nother:7b");
    }

    #[test]
    fn installed_is_smallest_first_with_sizes() {
        // /api/tags lists big first; the resource orders smallest-first, and the
        // json face carries the sizes hosts use for co-load budgeting.
        let mock = MockTransport::new(TAGS);
        let kernel = kernel_with(mock);
        let out = issue(
            &kernel,
            Request::new(
                Verb::Source,
                Iri::parse("urn:llm:ollama:installed").unwrap(),
            ),
        );
        assert_eq!(out.bytes, b"small:3b\nbig:latest");
        let json = issue(
            &kernel,
            Request::new(
                Verb::Source,
                Iri::parse("urn:llm:ollama:installed").unwrap(),
            )
            .with_arg("as", ArgRef::Inline(b"application/json".to_vec())),
        );
        let v: Value = serde_json::from_slice(&json.bytes).unwrap();
        assert_eq!(v[0]["model"], "small:3b");
        assert_eq!(v[0]["size"], 2000000000u64);
        assert_eq!(v[1]["model"], "big:latest");
        // MockTransport answers /api/show with the tags body — no capabilities
        // key — so the json face reports unknown as an empty array.
        assert_eq!(v[0]["capabilities"], json!([]));
    }

    #[test]
    fn an_explicit_model_is_never_substituted() {
        // model= was NAMED: a 404 must surface, not silently switch models.
        let mock = QueueTransport::new(vec![(404, NOT_FOUND)]);
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, OpenAiConfig::ollama("llama3.1"))));
        let denied = block_on(kernel.issue(
            ask("urn:llm:ollama:ask", "hi").with_arg("model", ArgRef::Inline(b"ghost".to_vec())),
            &Capability::root(),
        ));
        assert!(denied.is_err(), "explicit model + 404 must error");
        assert_eq!(mock.log.lock().unwrap().len(), 1, "no fallback attempted");
    }

    #[test]
    fn installed_lists_the_servers_models() {
        let mock = MockTransport::new(INSTALLED);
        let kernel = kernel_with(mock);
        let out = issue(
            &kernel,
            Request::new(
                Verb::Source,
                Iri::parse("urn:llm:ollama:installed").unwrap(),
            ),
        );
        assert_eq!(out.bytes, b"real:latest\nother:7b");
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
    fn model_reports_the_configured_id_verbatim() {
        // The identity face is a pure config read: it must answer with the
        // provider's default_model EXACTLY (no trailing newline, no quoting),
        // reach no network (DownTransport), need no net capability, and be
        // cacheable on the same grounds as urn:llm:config (registry loads at
        // host start; a restart is the only thing that changes it).
        let reg = Registry::from_json(TWO_PROVIDERS).unwrap();
        let kernel = Kernel::new(Arc::new(space(Arc::new(DownTransport), reg)));
        let no_caps = Capability::root().attenuate(Vec::<String>::new());
        let out = block_on(kernel.issue(
            Request::new(Verb::Source, Iri::parse("urn:llm:fast:model").unwrap()),
            &no_caps,
        ))
        .unwrap();
        assert_eq!(out.bytes, b"llama3.2:3b");
        assert_eq!(out.repr_type.canonical(), "text/plain;charset=utf-8");
        assert_eq!(
            out.expiry,
            ikigai_core::Expiry::Never,
            "config-derived: cacheable like urn:llm:config"
        );
        let remote = block_on(kernel.issue(
            Request::new(Verb::Source, Iri::parse("urn:llm:remote:model").unwrap()),
            &no_caps,
        ))
        .unwrap();
        assert_eq!(remote.bytes, b"gpt-4o");
    }

    // ---- a provider names the SERVER, the server names the model ------------

    /// What Brian's `rapid-mlx` really answers on `GET /v1/models` (trimmed to
    /// the fields we read). Note TWO entries for ONE set of weights — the
    /// canonical id and a lowercase alias — so "more than one model served" is
    /// the live case, not a hypothetical.
    const RAPID_MODELS: &str = r#"{"object":"list","data":[
        {"id":"mlx-community/Qwen3.8-27B-4bit","object":"model","owned_by":"rapid-mlx",
         "modality":"text","context_window":262144,"capabilities":["text","tools"]},
        {"id":"qwen3.8-27b-4bit","object":"model","owned_by":"rapid-mlx",
         "modality":"text","context_window":262144,"capabilities":["text","tools"]}]}"#;

    const CANONICAL: &str = "mlx-community/Qwen3.8-27B-4bit";

    /// A provider that pins nothing: `{ "base_url": … }`, governance only.
    const DISCOVERING: &str = r#"{ "default": "rapid", "providers": { "rapid": {
        "base_url": "http://localhost:8000/v1" } } }"#;

    fn registry_of(json: &str) -> Registry {
        Registry::from_json(json).unwrap()
    }

    #[test]
    fn from_json_accepts_an_entry_that_names_only_the_server() {
        let reg = registry_of(DISCOVERING);
        let rapid = reg
            .providers
            .iter()
            .find(|p| p.provider == "rapid")
            .unwrap();
        assert_eq!(rapid.default_model, None, "no model pinned");
        // ...and one that DOES pin a model still parses it verbatim.
        let pinned = registry_of(TWO_PROVIDERS);
        assert_eq!(
            pinned.providers[0].default_model.as_deref(),
            Some("llama3.2:3b")
        );
    }

    #[test]
    fn an_unpinned_provider_discovers_its_model_and_answers() {
        // GET the listing, then chat with what came back. Two models are served
        // (canonical + alias); the ordering rule resolves it rather than erroring.
        let mock = QueueTransport::new(vec![(200, RAPID_MODELS), (200, CANNED)]);
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, registry_of(DISCOVERING))));
        let out = issue(&kernel, ask("urn:llm:rapid:ask", "hi"));
        assert_eq!(out.bytes, b"Hello there!");

        let log = mock.log.lock().unwrap();
        assert_eq!(log.len(), 2, "one probe, one chat");
        assert_eq!(log[0].method, Method::Get);
        assert_eq!(log[0].url, "http://localhost:8000/v1/models");
        let body: Value = serde_json::from_slice(&log[1].body).unwrap();
        assert_eq!(body["model"], CANONICAL, "the discovered id, not a guess");
    }

    #[test]
    fn discovery_fills_capability_but_declared_values_win() {
        // The provider declares governance AND a context it wants to keep.
        let reg = registry_of(
            r#"{ "default": "rapid", "providers": { "rapid": {
                "base_url": "http://localhost:8000/v1",
                "caps": { "cost": "local", "vendor": "mlx", "context": 4096 } } } }"#,
        );
        let kernel = Kernel::new(Arc::new(space(MockTransport::new(RAPID_MODELS), reg)));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap()),
        );
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        let entry = &v["models"]["rapid"];
        assert_eq!(entry["model"], CANONICAL, "the model is discovered");
        assert_eq!(
            entry["caps"]["tools"], true,
            "tools discovered from listing"
        );
        assert_eq!(
            entry["caps"]["modalities"][0], "text",
            "modality discovered"
        );
        assert_eq!(
            entry["caps"]["context"], 4096,
            "a DECLARED context wins over the server's 262144"
        );
        assert_eq!(entry["caps"]["vendor"], "mlx", "declared vendor stands");
        assert_eq!(entry["caps"]["cost"], "local", "declared cost stands");
    }

    #[test]
    fn a_server_cannot_launder_itself_past_a_governance_exclusion() {
        // rapid-mlx self-reports `owned_by: "rapid-mlx"`, and the provider
        // declared no vendor. Governance is NOT discoverable: vendor stays
        // unstated, so `vendor!=openai` still fails it (it might BE openai).
        let kernel = Kernel::new(Arc::new(space(
            MockTransport::new(RAPID_MODELS),
            registry_of(DISCOVERING),
        )));
        let out = issue(
            &kernel,
            Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap()),
        );
        let v: Value = serde_json::from_slice(&out.bytes).unwrap();
        assert_eq!(v["models"]["rapid"]["caps"]["vendor"], Value::Null);
        assert_eq!(v["models"]["rapid"]["caps"]["cost"], Value::Null);
        // ...while the capability it DID advertise is there to route on.
        assert_eq!(v["models"]["rapid"]["caps"]["context"], 262144);

        let laundered = block_on(
            kernel.issue(
                Request::new(Verb::Source, Iri::parse("urn:llm:select").unwrap())
                    .with_arg("needs", ArgRef::Inline(b"vendor!=openai".to_vec())),
                &Capability::root(),
            ),
        );
        assert!(
            laundered.is_err(),
            "a self-reported owner must not satisfy a vendor exclusion"
        );
    }

    #[test]
    fn an_explicit_model_beats_discovery_and_probes_nothing() {
        // Naming a model is never a request to go looking for one.
        let mock = QueueTransport::new(vec![(200, CANNED)]);
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, registry_of(DISCOVERING))));
        let out = issue(
            &kernel,
            ask("urn:llm:rapid:ask", "hi").with_arg("model", ArgRef::Inline(b"named:7b".to_vec())),
        );
        assert_eq!(out.bytes, b"Hello there!");
        let log = mock.log.lock().unwrap();
        assert_eq!(log.len(), 1, "no probe: the caller named the model");
        let body: Value = serde_json::from_slice(&log[0].body).unwrap();
        assert_eq!(body["model"], "named:7b");
    }

    #[test]
    fn a_down_backend_names_the_base_url_and_substitutes_nothing() {
        let kernel = Kernel::new(Arc::new(space(
            Arc::new(DownTransport),
            registry_of(DISCOVERING),
        )));
        for iri in ["urn:llm:rapid:ask", "urn:llm:rapid:model"] {
            let failed = block_on(kernel.issue(ask(iri, "hi"), &Capability::root()));
            let msg = format!("{:?}", failed.unwrap_err());
            assert!(msg.contains("could not discover a model"), "{msg}");
            assert!(
                msg.contains("http://localhost:8000/v1"),
                "the error names the base_url: {msg}"
            );
        }
    }

    #[test]
    fn discovery_keeps_the_smallest_chat_capable_first_rule() {
        // An unpinned provider pointed at Ollama: native tags list big first,
        // the ordering rule sorts smallest-first, and the embedder that sorts
        // ahead of everything is passed over — the same rule the 404 fallback
        // uses, not a second one.
        let mock = QueueTransport::new(vec![
            (200, TAGS),
            (200, SHOW_EMBED), // small:3b
            (200, SHOW_CHAT),  // big:latest
            (200, CANNED),
        ]);
        let reg = registry_of(
            r#"{ "default": "any", "providers": { "any": {
                "base_url": "http://localhost:11434/v1",
                "caps": { "vendor": "ollama", "cost": "local" } } } }"#,
        );
        let transport: Arc<dyn HttpTransport> = mock.clone();
        let kernel = Kernel::new(Arc::new(space(transport, reg)));
        let out = issue(&kernel, ask("urn:llm:any:ask", "hi"));
        assert_eq!(out.bytes, b"Hello there!");
        let log = mock.log.lock().unwrap();
        let chat: Value = serde_json::from_slice(&log[3].body).unwrap();
        assert_eq!(chat["model"], "big:latest", "the embedder cannot chat");
    }

    #[test]
    fn the_identity_face_answers_from_config_or_from_the_server() {
        // PINNED: no network (DownTransport), no capability, permanently
        // cacheable, byte-identical to before. browse keys explain-archive
        // version tags on this — a changed id re-derives every explanation.
        let pinned = Kernel::new(Arc::new(space(
            Arc::new(DownTransport),
            registry_of(TWO_PROVIDERS),
        )));
        let no_caps = Capability::root().attenuate(Vec::<String>::new());
        let out = block_on(pinned.issue(
            Request::new(Verb::Source, Iri::parse("urn:llm:fast:model").unwrap()),
            &no_caps,
        ))
        .unwrap();
        assert_eq!(out.bytes, b"llama3.2:3b");
        assert_eq!(out.expiry, ikigai_core::Expiry::Never);

        // DISCOVERING: the discovered id is the only honest answer, it costs a
        // probe, and it is a LIVE fact — cacheing it would restore the staleness
        // discovery exists to remove.
        let mock = MockTransport::new(RAPID_MODELS);
        let discovering = Kernel::new(Arc::new(space(mock, registry_of(DISCOVERING))));
        let live = issue(
            &discovering,
            Request::new(Verb::Source, Iri::parse("urn:llm:rapid:model").unwrap()),
        );
        assert_eq!(String::from_utf8(live.bytes).unwrap(), CANONICAL);
        assert_eq!(live.expiry, ikigai_core::Expiry::Always, "uncacheable");

        // ...and it is capability-gated, because it reaches the network.
        let denied = block_on(discovering.issue(
            Request::new(Verb::Source, Iri::parse("urn:llm:rapid:model").unwrap()),
            &no_caps,
        ));
        assert!(denied.is_err(), "a probing :model needs urn:cap:net");
    }

    #[test]
    fn only_the_probing_identity_face_declares_the_net_capability() {
        // Declared = enforced, both directions: the config read must not
        // over-offer, the probe must not under-declare.
        let mock = MockTransport::new(CANNED);
        let pinned = ModelEndpoint::new(OpenAiConfig::ollama("llama3.1"), Arc::clone(&mock) as _);
        let discovering = ModelEndpoint::new(
            OpenAiConfig::discovering("rapid", "http://localhost:8000/v1"),
            Arc::clone(&mock) as _,
        );
        assert!(!pinned.describe().requires.iter().any(|c| c == CAP_NET));
        assert!(discovering.describe().requires.iter().any(|c| c == CAP_NET));
    }

    #[test]
    fn config_derived_resources_stay_cacheable_for_pinned_registries() {
        // The four pinned providers must pay nothing for this feature: a
        // registry with no discovering provider caches exactly as before.
        let pinned = Kernel::new(Arc::new(space(
            MockTransport::new(CANNED),
            registry_of(SELECTABLE),
        )));
        for (iri, arg) in [("urn:llm:models", None), ("urn:llm:select", Some("vision"))] {
            let mut req = Request::new(Verb::Source, Iri::parse(iri).unwrap());
            if let Some(needs) = arg {
                req = req.with_arg("needs", ArgRef::Inline(needs.as_bytes().to_vec()));
            }
            assert_eq!(
                issue(&pinned, req).expiry,
                ikigai_core::Expiry::Never,
                "{iri} is config-derived here"
            );
        }
        // One discovering provider makes the inventory a live fact.
        let live = Kernel::new(Arc::new(space(
            MockTransport::new(RAPID_MODELS),
            registry_of(DISCOVERING),
        )));
        assert_eq!(
            issue(
                &live,
                Request::new(Verb::Source, Iri::parse("urn:llm:models").unwrap())
            )
            .expiry,
            ikigai_core::Expiry::Always,
            "discovery must not be cached"
        );
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
