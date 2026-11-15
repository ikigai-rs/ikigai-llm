//! Baseline for the budget design spike; this test does not implement budgets.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Iri, Kernel, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig};

struct Unmetered(AtomicUsize);

#[async_trait]
impl HttpTransport for Unmetered {
    async fn send(&self, _: HttpRequest) -> Result<HttpResponse, String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: br#"{"choices":[{"message":{"content":"ok"}}]}"#.to_vec().into(),
        })
    }
}

#[test]
fn network_scope_alone_allows_repeated_asks_without_usage() {
    let transport = Arc::new(Unmetered(AtomicUsize::new(0)));
    let kernel = Kernel::new(Arc::new(space(
        transport.clone(),
        OpenAiConfig::ollama("m"),
    )));
    let capability = Capability::scoped(["urn:cap:net:localhost".to_string()]);
    for target in ["urn:llm:ask", "urn:llm:ollama:ask"] {
        let request = Request::new(Verb::Source, Iri::parse(target).unwrap())
            .with_arg("prompt", ArgRef::Inline(b"hello".to_vec()));
        let answer = futures::executor::block_on(kernel.issue(request, &capability)).unwrap();
        assert_eq!(answer.bytes, b"ok");
    }
    assert_eq!(transport.0.load(Ordering::SeqCst), 2);
}
