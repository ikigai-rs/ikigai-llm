//! `urn:llm:embed`: text to a vector from a NAMED model (ledger #3690).
//!
//! The facade (`urn:llm:embed`) routes like `urn:llm:ask` (`provider=`, else the
//! registry's default) to a per-provider backend (`urn:llm:<provider>:embed`). This
//! slice speaks Ollama's native `/api/embed`, so a backend is bound only for a provider
//! that declares `vendor: "ollama"`, the same opt-in that admits `/api/tags` and
//! `/api/show` discovery. An OpenAI-compatible `/v1/embeddings` names no digest, and
//! an answer that cannot name its digest is one a caller could compare against a
//! vector from another model.
//!
//! **The answer names the model and its digest.** Two vectors are comparable only
//! when they came from the same weights, so every answer carries the installed name
//! the server lists (`embeddinggemma` is answered as `embeddinggemma:latest`) and
//! the digest `/api/tags` gives for it. A model the server does not list is refused
//! as not found, naming the model, before any text is sent.
//!
//! **The text is a stranger's text.** It goes to the backend and nowhere else: it is
//! never written into an error (a server that echoes it has it cut, as a prompt is),
//! never noted in a trace, and never held by the kernel's cache, whose key is a
//! digest of the request (the text and the model among its arguments), not the text.
//!
//! **Local only, unless the configuration says otherwise.** A provider must declare
//! `cost: "local"`, or set `embed_remote: true` in its registry entry
//! ([`OpenAiConfig::embed_remote`]); otherwise the text is refused before anything
//! is sent. A provider that declares no cost is not known to be local.

use super::*;

/// The faces an embedding is served in: JSON only.
const JSON_ONLY: [&str; 1] = ["application/json"];

/// Whether `config` gets an embedding backend: Ollama's native API, opted into by a
/// declared `vendor: "ollama"`.
pub(crate) fn embeds(config: &OpenAiConfig) -> bool {
    config.caps.vendor.as_deref() == Some("ollama")
}

/// `urn:llm:embed`: picks a backend (`provider=`, else the configured default) and
/// re-issues the request to `urn:llm:<provider>:embed`.
pub struct EmbedFacade {
    registry: Registry,
}

impl EmbedFacade {
    /// A facade over `registry`, bound when at least one provider embeds.
    pub fn new(registry: Registry) -> Self {
        EmbedFacade { registry }
    }

    fn route(&self, inv: &Invocation<'_>) -> Result<Request> {
        let named = optional_str(inv, "provider")?;
        let provider = named.unwrap_or(&self.registry.default);
        let can_embed = self
            .registry
            .providers
            .iter()
            .any(|p| p.provider == provider && embeds(p));
        if !can_embed {
            let embedders: Vec<&str> = self
                .registry
                .providers
                .iter()
                .filter(|p| embeds(p))
                .map(|p| p.provider.as_str())
                .collect();
            let detail = format!(
                "provider `{provider}` does not embed text; providers that do: {} \
                 (an embedding backend is a provider declaring `vendor: \"ollama\"`)",
                embedders.join(", ")
            );
            return Err(match named {
                Some(_) => Error::InvalidArgument {
                    name: "provider".to_string(),
                    detail,
                },
                None => Error::Unsupported {
                    verb: None,
                    detail: format!("urn:llm:embed: the default {detail}; name one with provider="),
                },
            });
        }
        let target = Iri::parse(format!("urn:llm:{provider}:embed")).map_err(|e| {
            Error::Endpoint(format!("urn:llm:embed: bad provider `{provider}`: {e}"))
        })?;
        let mut req = Request::new(Verb::Source, target);
        req.args = inv.request.args.clone();
        Ok(req)
    }
}

#[async_trait]
impl Endpoint for EmbedFacade {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        inv.issue(self.route(inv)?).await
    }

    fn name(&self) -> &str {
        "llm-embed"
    }

    fn describe(&self) -> Description {
        let description = embed_description(
            "llm-embed",
            "Embed text: route to a backend (provider=, else the configured default) and \
             return the vector from the NAMED model, with the model's installed name and \
             digest, so vectors from different models are never compared. Cacheable.",
        )
        .input(
            ArgSpec::new("provider")
                .summary("backend to route to, e.g. ollama (default: configured)")
                .class(XSD_STRING)
                .optional(),
        );
        // The provider is chosen at invocation time: declare the parameterized floor,
        // and the routed backend checks the exact provider's use scope.
        if self
            .registry
            .providers
            .iter()
            .any(|p| embeds(p) && p.api_key_secret.is_some())
        {
            description.requires("urn:cap:llm:use:*")
        } else {
            description
        }
    }
}

/// `urn:llm:<provider>:embed`: one Ollama embedding backend over `/api/embed`.
pub struct EmbedBackend {
    /// `llm-<provider>-embed`: one id per bound backend.
    id: String,
    config: OpenAiConfig,
    transport: Arc<dyn HttpTransport>,
}

impl EmbedBackend {
    /// A backend for `config`, sending over `transport`.
    pub fn new(config: OpenAiConfig, transport: Arc<dyn HttpTransport>) -> Self {
        EmbedBackend {
            id: format!("llm-{}-embed", config.provider),
            config,
            transport,
        }
    }
}

#[async_trait]
impl Endpoint for EmbedBackend {
    async fn invoke(&self, inv: &Invocation<'_>) -> Result<Representation> {
        self.answer(inv).await
    }

    fn name(&self) -> &str {
        &self.id
    }

    fn describe(&self) -> Description {
        requires_use(
            embed_description(
                &self.id,
                "Embed text with a NAMED model on Ollama's /api/embed: the vector, the \
                 model's installed name, its digest and the dimensions. A model that is \
                 not installed is refused, naming it. Cacheable.",
            ),
            &self.config,
        )
    }
}

impl EmbedBackend {
    async fn answer(&self, inv: &Invocation<'_>) -> Result<Representation> {
        let provider = &self.config.provider;
        let who = format!("urn:llm:{provider}:embed");
        // Ollama's native API sits beside its OpenAI-compatible `/v1`.
        let root = self
            .config
            .base_url
            .trim_end_matches('/')
            .trim_end_matches("/v1");
        let tags_url = require_net(inv, root, "api/tags", &who)?;
        let embed_url = require_net(inv, root, "api/embed", &who)?;

        // Every input is read, and refused if unreadable, before anything is sent.
        let text = inv
            .inline_str("text")
            .or_else(|_| inv.inline_str("content"))
            .map_err(|_| Error::MissingArgument("text".to_string()))?;
        let named = optional_str(inv, "model")?
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| Error::MissingArgument("model".to_string()))?;
        requested_face(inv, &JSON_ONLY)?;

        if self.config.caps.cost.as_deref() != Some("local") && !self.config.embed_remote {
            return Err(Error::Denied(format!(
                "{who}: provider `{provider}` is not declared `cost: \"local\"`, and its \
                 configuration does not set `embed_remote: true`; text is embedded by a \
                 local model only unless the configuration allows a remote one"
            )));
        }

        let headers = provider_headers(inv, &self.config, true).await?;
        let (model, digest) = self.installed(inv, &tags_url, named, &who).await?;

        let payload = json!({ "model": model, "input": text });
        let response = post_json(self.transport.as_ref(), &embed_url, &headers, &payload).await?;
        // Labeled with the model and digest that answered; never the text.
        inv.trace_note("model", &model);
        inv.trace_note("digest", &digest);
        inv.trace_note("provider", provider);
        if let Some(cost) = &self.config.caps.cost {
            inv.trace_note("cost", cost);
        }
        if response.status >= 400 {
            return Err(status_failure(
                inv,
                &response,
                format!(
                    "{who}: embedding with `{model}` failed, the backend returned {}: {}",
                    response.status,
                    error_detail(&response, &[text])
                ),
            ));
        }
        let vector = serde_json::from_slice::<Value>(&response.body)
            .ok()
            .and_then(|v| one_vector(&v))
            .ok_or_else(|| {
                Error::Endpoint(format!(
                    "{who}: the backend's answer from /api/embed held no single vector of \
                     numbers for `{model}`; check that the server at `{}` is Ollama",
                    redacted_url(&self.config.base_url)
                ))
            })?;
        let answer = json!({
            "vector": vector,
            "model": model,
            "digest": digest,
            "dimensions": vector.len(),
        });
        // Deterministic for one model at one digest, and the answer names both, so a
        // copy served after a re-pull still says truthfully which weights made it. A
        // host that re-pulls cuts `urn:llm:<provider>:installed`; a registry reload
        // cuts `urn:llm:config`.
        Ok(Representation::new(
            ReprType::new("application/json").with_param("charset", "utf-8"),
            serde_json::to_vec(&answer).unwrap_or_default(),
        )
        .cacheable()
        .depends_on(CONFIG_THREAD)
        .depends_on(format!("urn:llm:{provider}:installed")))
    }

    /// The installed name and digest of the model the caller named, from Ollama's
    /// `/api/tags`. A name without a tag means `:latest`, as Ollama reads it. A model
    /// the server does not list is [`Error::NotFound`], naming it and the fix.
    async fn installed(
        &self,
        inv: &Invocation<'_>,
        url: &str,
        named: &str,
        who: &str,
    ) -> Result<(String, String)> {
        let response =
            receive(get(self.transport.as_ref(), inv, &self.config, url.to_string()).await?)
                .await?;
        if response.status >= 400 {
            return Err(status_failure(
                inv,
                &response,
                format!(
                    "{who}: listing the installed models (/api/tags) failed, the backend \
                     returned {}",
                    response.status
                ),
            ));
        }
        let listing: Value = serde_json::from_slice(&response.body).map_err(|_| {
            Error::Endpoint(format!(
                "{who}: the installed-model listing (/api/tags) was not JSON; check that \
                 the server at `{}` is Ollama",
                redacted_url(&self.config.base_url)
            ))
        })?;
        let root = redacted_url(url.trim_end_matches("/api/tags"));
        let tagged = with_tag(named);
        let entry = listing["models"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|m| {
                m["name"]
                    .as_str()
                    .or_else(|| m["model"].as_str())
                    .is_some_and(|name| name == named || name == tagged)
            })
            .ok_or_else(|| {
                Error::NotFound(format!(
                    "{who}: model `{named}` is not installed at `{root}`; install it with \
                     `ollama pull {named}`, or name one listed by urn:llm:{}:installed",
                    self.config.provider
                ))
            })?;
        let model = entry["name"]
            .as_str()
            .or_else(|| entry["model"].as_str())
            .unwrap_or(named)
            .to_string();
        let digest = entry["digest"]
            .as_str()
            .filter(|d| !d.is_empty())
            .ok_or_else(|| {
                Error::Endpoint(format!(
                    "{who}: the server lists `{model}` without a digest, so its vectors \
                     could not be told from another model's; nothing was embedded"
                ))
            })?
            .to_string();
        Ok((model, digest))
    }
}

/// `name` as Ollama reads it: with `:latest` when its last path segment has no tag.
fn with_tag(name: &str) -> String {
    let last = name.rsplit('/').next().unwrap_or(name);
    if last.contains(':') {
        name.to_string()
    } else {
        format!("{name}:latest")
    }
}

/// The one vector in an `/api/embed` answer: `embeddings` holding exactly one
/// non-empty array of finite numbers.
fn one_vector(v: &Value) -> Option<Vec<f64>> {
    let [only] = v["embeddings"].as_array()?.as_slice() else {
        return None;
    };
    let vector = only
        .as_array()?
        .iter()
        .map(|x| x.as_f64().filter(|x| x.is_finite()))
        .collect::<Option<Vec<f64>>>()?;
    (!vector.is_empty()).then_some(vector)
}

/// The shared parameter contract for the embedding facade and backends.
fn embed_description(id: &str, summary: &str) -> Description {
    let description = Description::new(id)
        .summary(summary)
        .verb(Verb::Source)
        .input(
            ArgSpec::new("text")
                .summary("the text to embed (or pipe it in as content)")
                .class(XSD_STRING),
        )
        .input(
            ArgSpec::new("model")
                .summary("the embedding model to use, e.g. embeddinggemma; there is no default")
                .class(XSD_STRING),
        )
        .input(
            ArgSpec::new("as")
                .summary("application/json: {vector, model, digest, dimensions}")
                .class(XSD_STRING)
                .one_of(JSON_ONLY)
                .default_value(JSON_ONLY[0])
                .optional(),
        )
        .output("application/json;charset=utf-8")
        .requires(CAP_NET);
    fails_like_an_origin(description)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_without_a_tag_means_latest() {
        assert_eq!(with_tag("embeddinggemma"), "embeddinggemma:latest");
        assert_eq!(with_tag("embeddinggemma:300m"), "embeddinggemma:300m");
        assert_eq!(with_tag("hf.co/org/model"), "hf.co/org/model:latest");
        assert_eq!(
            with_tag("registry.local:5000/ns/model"),
            "registry.local:5000/ns/model:latest"
        );
    }

    #[test]
    fn only_one_finite_vector_is_an_answer() {
        assert_eq!(
            one_vector(&json!({ "embeddings": [[0.5, -1]] })),
            Some(vec![0.5, -1.0])
        );
        for bad in [
            json!({}),
            json!({ "embeddings": [] }),
            json!({ "embeddings": [[]] }),
            json!({ "embeddings": [[1.0], [2.0]] }),
            json!({ "embeddings": [[1.0, "x"]] }),
            json!({ "embedding": [1.0] }),
        ] {
            assert_eq!(one_vector(&bad), None, "{bad}");
        }
    }
}
