//! Live smoke test against a real OpenAI-compatible server (default: local Ollama).
//!
//! Start Ollama and pull a model, then:
//!
//! ```text
//! MODEL=llama3.1 cargo run --example ask -- "Explain resource-oriented computing in one sentence."
//! ```
//!
//! This is the one thing the mocks can't prove: that the backend actually speaks
//! to a real server. It uses a blocking `ureq` transport so it needs no async
//! runtime and touches nothing in the shipped crate.

use std::io::Read;
use std::sync::Arc;

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig};

/// A minimal blocking transport — exactly the seam a native host fills with
/// `ureq`/`reqwest` (the browser would fill it with `fetch`).
struct UreqTransport;

#[async_trait]
impl HttpTransport for UreqTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let mut req = ureq::request(request.method.as_str(), &request.url);
        for (k, v) in &request.headers {
            req = req.set(k, v);
        }
        let (status, mut reader) = match req.send_bytes(&request.body) {
            Ok(resp) => (resp.status(), resp.into_reader()),
            Err(ureq::Error::Status(code, resp)) => (code, resp.into_reader()),
            Err(e) => return Err(e.to_string()),
        };
        let mut body = Vec::new();
        reader.read_to_end(&mut body).map_err(|e| e.to_string())?;
        Ok(HttpResponse {
            status,
            headers: vec![],
            body,
        })
    }
}

fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello in exactly five words.".to_string());
    let model = std::env::var("MODEL").unwrap_or_else(|_| "llama3.1".to_string());

    let kernel = Kernel::new(Arc::new(space(
        Arc::new(UreqTransport),
        OpenAiConfig::ollama(&model),
    )));

    // urn:llm:ask (the facade) → urn:llm:ollama:ask (the backend) → the server.
    let request = Request::new(Verb::Source, Iri::parse("urn:llm:ask").unwrap())
        .with_arg("prompt", ArgRef::Inline(prompt.clone().into_bytes()));

    println!("model:  {model}\nprompt: {prompt}\n---");
    match futures::executor::block_on(kernel.issue(request, &Capability::root())) {
        Ok(repr) => println!("{}", String::from_utf8_lossy(&repr.bytes)),
        Err(e) => eprintln!("error: {e:?}"),
    }
}
