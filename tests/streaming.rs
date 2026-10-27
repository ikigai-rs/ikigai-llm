//! Exercise the live body at the transport seam; no inference server or timer.
use std::collections::VecDeque;
use std::io;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::{executor::block_on, io::AsyncRead, StreamExt};
use ikigai_core::{
    ArgRef, Bindings, Body, Capability, Endpoint, Error, FailureClass, Invocation, Iri, Kernel,
    Request, Verb,
};
use ikigai_http::{HttpBody, HttpRequest, HttpResponse, HttpTransport, DEFAULT_MAX_RESPONSE_BYTES};
use ikigai_llm::{space, OpenAiBackend, OpenAiConfig};

const FIRST: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
const SECOND: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\" there!\"}}]}\n\n";
const DONE: &[u8] = b"data: [DONE]\n\n";
const BUFFERED: &[u8] = br#"{"model":"tiny","choices":[{"message":{"content":"Hello there!"},"finish_reason":"stop"}],"usage":{"total_tokens":4}}"#;

struct Chunks {
    reads: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    chunks: VecDeque<Vec<u8>>,
    repeat: Option<Vec<u8>>,
    fail: bool,
}
impl Drop for Chunks {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl AsyncRead for Chunks {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        out: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if out.is_empty() {
            return Poll::Ready(Ok(0));
        }
        if self.chunks.is_empty() {
            if let Some(repeat) = self.repeat.clone() {
                self.chunks.push_back(repeat);
            }
        }
        let Some(chunk) = self.chunks.front_mut() else {
            return Poll::Ready(if self.fail {
                Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "fixture reset",
                ))
            } else {
                Ok(0)
            });
        };
        let n = out.len().min(chunk.len());
        out[..n].copy_from_slice(&chunk[..n]);
        chunk.drain(..n);
        if chunk.is_empty() {
            self.chunks.pop_front();
        }
        Poll::Ready(Ok(n))
    }
}
struct Streaming {
    reads: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    chunks: Vec<Vec<u8>>,
    repeat: Option<Vec<u8>>,
    fail: bool,
    status: u16,
}
impl Streaming {
    fn new(chunks: &[&[u8]]) -> Self {
        Self {
            reads: Arc::new(AtomicUsize::new(0)),
            drops: Arc::new(AtomicUsize::new(0)),
            chunks: chunks.iter().map(|c| c.to_vec()).collect(),
            repeat: None,
            fail: false,
            status: 200,
        }
    }
}
#[async_trait]
impl HttpTransport for Streaming {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let payload: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        if payload["stream"] == false {
            return Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".into(), "application/json".into())],
                body: BUFFERED.to_vec().into(),
            });
        }
        assert_eq!(
            payload["stream"], true,
            "the body path must request streaming"
        );
        Ok(HttpResponse {
            status: self.status,
            headers: vec![(
                "content-type".into(),
                "text/event-stream; charset=utf-8".into(),
            )],
            body: HttpBody::from_reader(Chunks {
                reads: self.reads.clone(),
                drops: self.drops.clone(),
                chunks: self.chunks.clone().into(),
                repeat: self.repeat.clone(),
                fail: self.fail,
            }),
        })
    }
}
fn request(target: &str) -> Request {
    Request::new(Verb::Source, Iri::parse(target).unwrap())
        .with_arg("prompt", ArgRef::Inline(b"hi".to_vec()))
}
fn kernel(transport: Arc<Streaming>) -> Kernel {
    Kernel::new(Arc::new(space(transport, OpenAiConfig::ollama("tiny"))))
}
#[test]
fn tokens_are_polled_incrementally_through_the_facade_and_backend() {
    for provider in ["ollama", "mlx"] {
        for target in ["urn:llm:ask".to_string(), format!("urn:llm:{provider}:ask")] {
            let transport = Arc::new(Streaming::new(&[FIRST, SECOND, DONE]));
            let mut config = OpenAiConfig::ollama("tiny");
            config.provider = provider.into();
            if provider == "mlx" {
                config.base_url = "http://localhost:8000/v1".into();
            }
            let kernel = Kernel::new(Arc::new(space(transport.clone(), config)));
            let answer =
                block_on(kernel.issue_body(request(&target), &Capability::root())).unwrap();
            assert!(matches!(answer.bytes, Body::Once(_)));
            assert_eq!(
                transport.reads.load(Ordering::SeqCst),
                0,
                "return before reading any tokens"
            );
            let mut stream = answer.bytes.into_stream();
            assert_eq!(block_on(stream.next()).unwrap().unwrap(), "Hello");
            assert_eq!(
                transport.reads.load(Ordering::SeqCst),
                1,
                "do not read ahead to completion"
            );
            drop(stream);
            assert_eq!(
                transport.drops.load(Ordering::SeqCst),
                1,
                "dropping cancels production"
            );
        }
    }
}

#[test]
fn collected_body_equals_buffered_invoke_and_kernel_bytes() {
    let transport = Arc::new(Streaming::new(&[FIRST, SECOND, DONE]));
    let backend = OpenAiBackend::new(OpenAiConfig::ollama("tiny"), transport.clone());
    let req = request("urn:llm:ollama:ask");
    let (bindings, cap) = (Bindings::new(), Capability::root());
    let inv = Invocation::detached(&req, &bindings, &cap);
    let buffered = block_on(backend.invoke(&inv)).unwrap();
    let streamed = block_on(backend.invoke_body(&inv)).unwrap();
    assert!(streamed.bytes.is_one_shot());
    assert_eq!(
        block_on(streamed.collect(1024)).unwrap().bytes,
        buffered.bytes
    );
    let kernel = kernel(transport);
    assert_eq!(
        block_on(kernel.issue(request("urn:llm:ask"), &cap))
            .unwrap()
            .bytes,
        buffered.bytes
    );
    // The normalized JSON face keeps its existing buffered request and envelope.
    let req = req.with_arg("as", ArgRef::Inline(b"application/json".to_vec()));
    let inv = Invocation::detached(&req, &bindings, &cap);
    let a = block_on(backend.invoke(&inv)).unwrap();
    let b = block_on(backend.invoke_body(&inv)).unwrap();
    assert_eq!(block_on(b.collect(1024)).unwrap().bytes, a.bytes);
}
#[test]
fn a_done_marker_is_required_and_failures_are_terminal() {
    for (tail, fail, class) in [
        (b"".as_slice(), false, FailureClass::Transient),
        (
            b"data: {bad JSON}\n\n".as_slice(),
            false,
            FailureClass::Unknown,
        ),
        (
            b"data: {\"error\":\"hi secret\"}\n\n".as_slice(),
            false,
            FailureClass::Unknown,
        ),
        (b"data: \xff\n\n".as_slice(), false, FailureClass::Unknown),
        (
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".as_slice(),
            false,
            FailureClass::Transient,
        ),
        (
            b"data: {\"choices\":[{\"delta\":{\"refusal\":\"hi secret\"}}]}\n\n".as_slice(),
            false,
            FailureClass::Refused,
        ),
        (
            b"data: {\"choices\":[{}]}\n\n".as_slice(),
            false,
            FailureClass::Unknown,
        ),
        (b"".as_slice(), true, FailureClass::Transient),
    ] {
        let mut fixture = Streaming::new(&[FIRST]);
        if !tail.is_empty() {
            fixture.chunks.push(tail.to_vec());
        }
        fixture.fail = fail;
        let transport = Arc::new(fixture);
        let description =
            OpenAiBackend::new(OpenAiConfig::ollama("tiny"), transport.clone()).describe();
        let kernel = kernel(transport);
        let answer =
            block_on(kernel.issue_body(request("urn:llm:ask"), &Capability::root())).unwrap();
        let mut stream = answer.bytes.into_stream();
        assert_eq!(block_on(stream.next()).unwrap().unwrap(), "Hello");
        let err = loop {
            match block_on(stream.next()).expect("a terminal error before EOF") {
                Ok(bytes) => assert!(bytes.is_empty(), "only metadata may precede the error"),
                Err(error) => break error,
            }
        };
        assert_eq!(err.class(), class);
        assert_eq!(
            description.action_specs()[0].admits_failure(class),
            Some(true)
        );
        assert!(!err.to_string().contains("secret"));
        assert!(
            block_on(stream.next()).is_none(),
            "one terminal error, no successful tail"
        );
    }
}
#[test]
fn done_stops_without_polling_a_broken_connection_after_it() {
    let mut fixture = Streaming::new(&[FIRST, DONE]);
    fixture.fail = true;
    let transport = Arc::new(fixture);
    let kernel = kernel(transport.clone());
    let result = block_on(kernel.issue(request("urn:llm:ask"), &Capability::root())).unwrap();
    assert_eq!(result.bytes, b"Hello");
    assert_eq!(transport.reads.load(Ordering::SeqCst), 2);
}
#[test]
fn sse_handles_fragmented_utf8_crlf_comments_and_metadata() {
    let wire = concat!(
        ": comment\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\r\n\r\n",
        "data: {\"choices\":\r\ndata: [{\"delta\":{\"content\":\"hé🌍\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n",
        "data: {\"choices\":[],\"usage\":{\"total_tokens\":4}}\r\n\r\n",
        "data: [DONE]\r\n\r\n"
    );
    let mut fixture = Streaming::new(&[]);
    fixture.chunks = wire.as_bytes().iter().map(|b| vec![*b]).collect();
    let kernel = kernel(Arc::new(fixture));
    assert_eq!(
        block_on(kernel.issue(request("urn:llm:ask"), &Capability::root()))
            .unwrap()
            .bytes,
        "hé🌍".as_bytes()
    );
}
#[test]
fn a_small_byte_budget_refuses_collection_and_stops_the_producer() {
    let mut fixture = Streaming::new(&[FIRST]);
    fixture.repeat = Some(SECOND.to_vec()); // infinitely large answer, never held whole
    let transport = Arc::new(fixture);
    let kernel = kernel(transport.clone()).with_stream_memory(8, 8);
    assert!(matches!(
        block_on(kernel.issue(request("urn:llm:ask"), &Capability::root())),
        Err(Error::OverBudget { .. })
    ));
    assert_eq!(transport.reads.load(Ordering::SeqCst), 2);
    assert_eq!(transport.drops.load(Ordering::SeqCst), 1);
}
#[test]
fn a_live_feed_can_exceed_the_buffered_wire_limit_without_accumulating_it() {
    let mut fixture = Streaming::new(&[]);
    fixture.repeat = Some(FIRST.to_vec());
    let transport = Arc::new(fixture);
    let kernel = kernel(transport.clone());
    let answer = block_on(kernel.issue_body(request("urn:llm:ask"), &Capability::root())).unwrap();
    let mut stream = answer.bytes.into_stream();
    let events = DEFAULT_MAX_RESPONSE_BYTES / FIRST.len() + 1;
    for _ in 0..events {
        assert_eq!(block_on(stream.next()).unwrap().unwrap(), "Hello");
    }
    assert_eq!(transport.reads.load(Ordering::SeqCst), events);
    drop(stream);
    assert_eq!(transport.drops.load(Ordering::SeqCst), 1);
}
#[test]
fn an_unterminated_event_is_bounded_and_refused() {
    let mut fixture = Streaming::new(&[b"data: "]);
    fixture.repeat = Some(vec![b'x'; 8192]);
    let transport = Arc::new(fixture);
    let kernel = kernel(transport.clone());
    let answer = block_on(kernel.issue_body(request("urn:llm:ask"), &Capability::root())).unwrap();
    assert!(matches!(
        block_on(answer.collect(1024)),
        Err(Error::OverBudget { .. })
    ));
    assert!(transport.reads.load(Ordering::SeqCst) <= 64 * 1024 / 8192 + 2);
}

struct TestClock(AtomicUsize);
impl ikigai_core::Clock for TestClock {
    fn now(&self) -> ikigai_core::Time {
        ikigai_core::Time::from_millis(self.0.load(Ordering::SeqCst) as u64)
    }
}
#[test]
fn the_run_deadline_refuses_the_next_token_and_drops_the_provider() {
    for tail in [
        SECOND,
        DONE,
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".as_slice(),
    ] {
        let transport = Arc::new(Streaming::new(&[FIRST, tail, DONE]));
        let clock = Arc::new(TestClock(AtomicUsize::new(100)));
        let kernel = kernel(transport.clone()).with_clock(clock.clone());
        let context =
            ikigai_core::IssueContext::new().with_not_after(ikigai_core::Time::from_millis(200));
        let answer = block_on(kernel.issue_body_with_context(
            request("urn:llm:ask"),
            &Capability::root(),
            context,
        ))
        .unwrap();
        let mut stream = answer.bytes.into_stream();
        assert_eq!(block_on(stream.next()).unwrap().unwrap(), "Hello");
        clock.0.store(200, Ordering::SeqCst);
        assert!(matches!(
            block_on(stream.next()).unwrap(),
            Err(Error::OverBudget { .. })
        ));
        assert!(block_on(stream.next()).is_none());
        assert_eq!(transport.drops.load(Ordering::SeqCst), 1);
    }
}
