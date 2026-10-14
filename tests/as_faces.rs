//! `as=` is matched the way the manifold declares it (ledger #179).
//!
//! Every endpoint here declares its `as` input as a `one_of` set of media types,
//! and `urn:kernel:validate` holds a proposed call to that set. The code used to
//! test `as.contains("json")` (or `"turtle"`) instead, so the two disagreed:
//! `as=application/ld+json` was refused by the manifold and answered with JSON by
//! the code, `as=APPLICATION/JSON` (media types are case-insensitive) silently
//! got text, and an undeclared face (`as=text/turtle` on `:ask`) silently got the
//! default. Now the code reads the media type's ESSENCE (parameters dropped,
//! case folded) and matches it against the same declared set: a declared face is
//! served, and anything else is refused, as the manifold refuses it.
//!
//! No real model is called: [`Stub`] answers every request in-process.

use std::sync::Arc;

use async_trait::async_trait;
use ikigai_core::{ArgRef, Capability, Error, Iri, Kernel, Representation, Request, Verb};
use ikigai_http::{HttpRequest, HttpResponse, HttpTransport};
use ikigai_llm::{space, OpenAiConfig, Registry};

/// Answers a chat completion, and a model listing, whatever was asked.
struct Stub;

#[async_trait]
impl HttpTransport for Stub {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, String> {
        let body = if request.url.ends_with("/models") {
            r#"{"data":[{"id":"m"}]}"#
        } else {
            r#"{"model":"m","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}"#
        };
        Ok(HttpResponse {
            status: 200,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body.as_bytes().to_vec(),
        })
    }
}

fn kernel() -> Kernel {
    // A provider with no vendor, so no Ollama-native discovery is attempted.
    let mut config = OpenAiConfig::discovering("p", "http://127.0.0.1:9/v1");
    config.default_model = Some("m".into());
    Kernel::new(Arc::new(space(Arc::new(Stub), Registry::single(config))))
}

fn source(target: &str, args: &[(&str, &str)]) -> Result<Representation, Error> {
    let mut request = Request::new(Verb::Source, Iri::parse(target).unwrap());
    for (name, value) in args {
        request = request.with_arg(*name, ArgRef::Inline(value.as_bytes().to_vec()));
    }
    futures::executor::block_on(kernel().issue(request, &Capability::root()))
}

/// The media type's essence the endpoint answered with.
fn served(target: &str, args: &[(&str, &str)]) -> String {
    let repr = source(target, args).unwrap_or_else(|e| panic!("{target} {args:?}: {e}"));
    repr.repr_type.media_type.clone()
}

fn refused(target: &str, args: &[(&str, &str)]) -> Error {
    match source(target, args) {
        Ok(repr) => panic!(
            "{target} {args:?} must be refused, but answered {}",
            repr.repr_type.media_type
        ),
        Err(e) => e,
    }
}

/// Each endpoint with an `as` input, a value to satisfy any other required
/// input, and its declared faces (default first).
/// An endpoint IRI, the arguments it needs besides `as`, and its faces.
type Case = (
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static [&'static str],
);

const ENDPOINTS: &[Case] = &[
    (
        "urn:llm:p:ask",
        &[("prompt", "hi")],
        &["text/plain", "application/json"],
    ),
    (
        "urn:llm:ask",
        &[("prompt", "hi")],
        &["text/plain", "application/json"],
    ),
    ("urn:llm:models", &[], &["application/json", "text/turtle"]),
    (
        "urn:llm:select",
        &[("needs", "provider=p")],
        &["text/plain", "application/json"],
    ),
    (
        "urn:llm:p:installed",
        &[],
        &["text/plain", "application/json"],
    ),
];

fn with<'a>(base: &[(&'a str, &'a str)], face: &'a str) -> Vec<(&'a str, &'a str)> {
    let mut args = base.to_vec();
    args.push(("as", face));
    args
}

#[test]
fn every_declared_face_is_served_and_absent_means_the_default() {
    for (target, base, faces) in ENDPOINTS {
        assert_eq!(served(target, base), faces[0], "{target}: the default");
        for face in *faces {
            assert_eq!(served(target, &with(base, face)), *face, "{target}");
        }
    }
}

/// Parameters are not part of the match, and the essence is case-insensitive
/// (RFC 9110 section 8.3.1).
#[test]
fn parameters_and_case_do_not_change_the_face() {
    for (target, base, faces) in ENDPOINTS {
        for face in *faces {
            for spelled in [
                format!("{face};charset=utf-8"),
                format!("{face} ; charset=UTF-8"),
                face.to_ascii_uppercase(),
            ] {
                assert_eq!(
                    served(target, &with(base, &spelled)),
                    *face,
                    "{target} as={spelled}"
                );
            }
        }
    }
}

/// A face the endpoint does not declare is refused, never answered with
/// something else: `contains("json")` served JSON for `application/ld+json`,
/// and an unknown face silently got the default.
#[test]
fn an_undeclared_face_is_refused() {
    for (target, base, faces) in ENDPOINTS {
        for face in [
            "application/ld+json",
            "text/json",
            "json",
            "text/turtle",
            "text/plain",
            "application/json",
            "",
        ] {
            if faces.contains(&face) {
                continue;
            }
            let err = refused(target, &with(base, face));
            assert!(
                matches!(&err, Error::InvalidArgument { name, .. } if name == "as"),
                "{target} as={face}: {err:?}"
            );
            for declared in *faces {
                assert!(err.to_string().contains(declared), "names the faces: {err}");
            }
        }
    }
}

/// And the manifold agrees: what the code refuses, `urn:kernel:validate`
/// refuses, and every declared face conforms.
#[test]
fn the_manifold_agrees_on_what_is_accepted() {
    let validate = |target: &str, face: &str| -> bool {
        let report = source(
            "urn:kernel:validate",
            &[
                ("endpoint", target),
                ("verb", "source"),
                ("args", &format!("prompt=hi&as={face}")),
            ],
        )
        .expect("validate answers");
        let report = String::from_utf8_lossy(&report.bytes).into_owned();
        !report.contains("is not an accepted value of `as`")
    };
    for face in ["text/plain", "application/json"] {
        assert!(validate("urn:llm:p:ask", face), "{face}");
        assert!(source("urn:llm:p:ask", &[("prompt", "hi"), ("as", face)]).is_ok());
    }
    for face in ["application/ld+json", "text/json", "text/turtle"] {
        assert!(!validate("urn:llm:p:ask", face), "{face}");
        assert!(source("urn:llm:p:ask", &[("prompt", "hi"), ("as", face)]).is_err());
    }
}

/// A transport that counts what was sent and answers nothing useful.
struct Counting(Arc<std::sync::atomic::AtomicUsize>);

#[async_trait]
impl HttpTransport for Counting {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, String> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err("unreachable".to_string())
    }
}

/// An undeclared face is refused before anything is sent: a prompt does not
/// leave for a backend whose answer could not be served.
#[test]
fn an_undeclared_face_sends_nothing() {
    let sent = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut config = OpenAiConfig::discovering("p", "http://127.0.0.1:9/v1");
    config.default_model = Some("m".into());
    let kernel = Kernel::new(Arc::new(space(
        Arc::new(Counting(Arc::clone(&sent))),
        Registry::single(config),
    )));
    for (target, base, _) in ENDPOINTS {
        let mut request = Request::new(Verb::Source, Iri::parse(*target).unwrap());
        for (name, value) in with(base, "text/html") {
            request = request.with_arg(name, ArgRef::Inline(value.as_bytes().to_vec()));
        }
        let err = futures::executor::block_on(kernel.issue(request, &Capability::root()))
            .expect_err("refused");
        assert!(
            matches!(err, Error::InvalidArgument { .. }),
            "{target}: {err:?}"
        );
    }
    assert_eq!(sent.load(std::sync::atomic::Ordering::SeqCst), 0);
}
