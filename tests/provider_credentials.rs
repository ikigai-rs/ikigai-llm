//! A provider's own credentials never reach the error a caller sees (ledger #4020).
//!
//! An error body's reason fields pass into the error by design (see
//! `tests/error_bodies.rs`), and a provider may write the credentials it was sent
//! into that reason: OpenAI's 401 says "Incorrect API key provided: <key>", and a
//! proxy or a self-hosted server may quote the whole `Authorization` value or the
//! request URL. That error travels through traces, logs, MCP replies and the far
//! side of a mount, to callers who hold only `urn:llm:*`. So the bearer token (on
//! its own and as `Bearer <token>`), a `base_url`'s `user:password`, and a key read
//! through `api_key_secret` are each replaced with `[credential]` in the error of
//! `ask` and of `embed`, while the rest of the reason still reaches the caller.
//!
//! No real model is called: [`Echo`] is an in-process [`HttpTransport`] whose error
//! answers quote what it was sent.

use std::sync::Arc;

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport, Method};
use ikigai_llm::{space, space_with_secrets, OpenAiConfig, Registry, SecretReader};
use serde_json::json;

const KEY: &str = "sk-PROVIDERKEY-7f3a91";
const PASSWORD: &str = "pr0xy-pass-55c1";
const BASE_URL: &str = "http://proxyuser:pr0xy-pass-55c1@localhost:11434/v1";

/// A provider whose error answers quote the credentials it was sent: the whole
/// `Authorization` value, the token on its own, and the request URL. `/api/tags`
/// answers 200 listing `m:latest`, so `embed` reaches `/api/embed`.
struct Echo {
    status: u16,
}

#[async_trait]
impl HttpTransport for Echo {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let json_reply = |status: u16, body: serde_json::Value| HttpResponse {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body.to_string().into_bytes().into(),
        };
        if request.method == Method::Get && request.url.ends_with("/api/tags") {
            return Ok(json_reply(
                200,
                json!({ "models": [{ "name": "m:latest", "size": 1u64, "digest": "d1" }] }),
            ));
        }
        let auth = request
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        let token = auth.strip_prefix("Bearer ").unwrap_or_default();
        let message = format!(
            "Incorrect API key provided: {auth}; token {token}; url {}",
            request.url
        );
        Ok(json_reply(
            self.status,
            json!({ "error": { "message": message } }),
        ))
    }
}

struct OneSecret;

impl SecretReader for OneSecret {
    fn read(&self, _name: &str) -> ikigai_core::Result<Vec<u8>> {
        Ok(format!("{KEY}\n").into_bytes())
    }
}

fn keyed() -> OpenAiConfig {
    OpenAiConfig {
        api_key: Some(KEY.into()),
        base_url: BASE_URL.into(),
        ..OpenAiConfig::ollama("m")
    }
}

fn kernel(status: u16, config: OpenAiConfig) -> Kernel {
    Kernel::new(Arc::new(space(
        Arc::new(Echo { status }),
        Registry::single(config),
    )))
}

fn error_of(kernel: &Kernel, target: &str, args: &[(&str, &str)]) -> String {
    let mut request = Request::new(Verb::Source, Iri::parse(target).unwrap());
    for (name, value) in args {
        request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
    }
    let err = futures::executor::block_on(kernel.issue(request, &Capability::root()))
        .expect_err("the provider answered an error status");
    format!("{err} / {err:?}")
}

fn assert_no_credentials(error: &str) {
    assert!(!error.contains("PROVIDERKEY"), "the key is in: {error}");
    assert!(!error.contains(PASSWORD), "the URL password is in: {error}");
    assert!(
        error.contains("[credential]"),
        "nothing marks the cut: {error}"
    );
    assert!(
        error.contains("Incorrect API key provided"),
        "the provider's reason is still reported: {error}"
    );
}

const ASK: &[(&str, &str)] = &[("prompt", "hi"), ("model", "m")];
const EMBED: &[(&str, &str)] = &[("text", "hi"), ("model", "m")];

#[test]
fn an_ask_error_quoting_the_key_and_url_reports_neither() {
    for status in [400, 401, 500] {
        let error = error_of(&kernel(status, keyed()), "urn:llm:ollama:ask", ASK);
        assert_no_credentials(&error);
    }
}

#[test]
fn an_embed_error_quoting_the_key_and_url_reports_neither() {
    for status in [400, 401, 500] {
        let error = error_of(&kernel(status, keyed()), "urn:llm:ollama:embed", EMBED);
        assert_no_credentials(&error);
    }
}

#[test]
fn a_key_read_through_api_key_secret_is_cut_from_the_error_too() {
    let config = OpenAiConfig {
        api_key: None,
        api_key_secret: Some("provider-key".into()),
        base_url: BASE_URL.into(),
        ..OpenAiConfig::ollama("m")
    };
    for target in ["ask", "embed"] {
        let space = space_with_secrets(
            Arc::new(Echo { status: 401 }),
            Arc::new(OneSecret),
            Registry::single(config.clone()),
        );
        let kernel = Kernel::new(Arc::new(space));
        let args = if target == "ask" { ASK } else { EMBED };
        let error = error_of(&kernel, &format!("urn:llm:ollama:{target}"), args);
        assert_no_credentials(&error);
    }
}

#[test]
fn an_unkeyed_provider_keeps_its_reason_whole() {
    let config = OpenAiConfig::ollama("m");
    let error = error_of(&kernel(401, config), "urn:llm:ollama:ask", ASK);
    assert!(!error.contains("[credential]"), "{error}");
    assert!(
        error.contains("url http://localhost:11434/v1/chat/completions"),
        "{error}"
    );
}
