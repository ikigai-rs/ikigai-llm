# Accessibility

What `ikigai-llm` emits that people read, what of it is this repository's and what is a model's,
the rules its own text follows, how it is tested, and how to report a barrier. It is not a
conformance claim. Adapted from the
[ACCESSIBILITY.md template](https://github.com/mgifford/ACCESSIBILITY.md) (MIT).

| Field | Value |
|---|---|
| Target | WCAG 2.2 Level AA, for any page that presents this output; this repository writes text only (plain text, JSON and Turtle), no HTML or SVG |
| Status | Evaluation in progress. No conformance claim. |
| Owner | the maintainers of this repository |
| Report a barrier | [a GitHub issue](https://github.com/ikigai-rs/ikigai-llm/issues/new?labels=accessibility) with the `accessibility` label; anything security-sensitive through [private vulnerability reporting](https://github.com/ikigai-rs/ikigai-llm/security/advisories/new) |
| Last reviewed | 2026-11-12 |

## What it emits that people read

- **A model's answer** from `urn:llm:ask` and `urn:llm:<provider>:ask` (`src/lib.rs`, streamed
  through `src/streaming.rs`): as `text/plain`, the completion text exactly as the model wrote
  it; as `application/json`, an envelope of that text, the model, the finish reason and usage.
- **Its own short answers**: the model id (`urn:llm:<provider>:model`), `true` or `false`
  (`urn:llm:<provider>:up`), a newline list of installed models
  (`urn:llm:<provider>:installed`), the winning backend's IRI (`urn:llm:select`), and the
  registry and profiles as JSON or Turtle (`urn:llm:config`, `urn:llm:models`).
- **Refusals** (`src/lib.rs`): a config that does not hold (`urn:llm:config: provider … declares
  …`), an input outside its form (`as`, `temperature`, `max_tokens`, `needs`), no backend that
  satisfies `needs=`, a host the capability does not reach, a model that could not be
  discovered, a backend's failure typed by status, and a backend's redirect, which names the
  status and the `Location` and is never followed.
- **`examples/ask.rs`**, which prints the model, the prompt, and the answer or the error.
- **Documentation**: `README.md`, `AGENTS.md` and this statement are Markdown, rendered by
  GitHub or read in an editor; Rust API documentation is generated from source comments.

**The completion text is the model's, not this repository's.** This crate does not change,
check, label or describe it: whatever the model writes (Markdown, code, emoji, another language,
a wrong or harmful answer) reaches the caller as written. A host that shows it to a person owns
how it is presented. Also not this repository's: the sentence each error kind is wrapped in
(ikigai-core's `Display`), and the reason field of a backend's JSON error body, which is passed
through with the caller's prompt cut out.

## The rules that output follows

- A refusal names the input or setting at fault and what is accepted: `as` lists the faces the
  endpoint serves, `temperature` and `max_tokens` say the form expected, a config error names the
  provider and the fix (`keep only api_key_secret`), no backend for `needs=` lists the providers
  and points to `urn:llm:models`, and a denial names the host and the capability it needs.
- A backend's redirect names its status and its `Location` (with any `user:password@` removed)
  and says the fix (`set base_url to where the provider is`); it is never followed.
- A backend's error page or plain-text refusal is described (status, media type, length), never
  forwarded, and a URL's `user:password@` is removed from error text.
- No color is written; meaning is in words.

## How it is tested

Every named check has a result: Pass, Fail or Blocked, with evidence. A Pass covers only
that check, not overall accessibility or WCAG conformance. The checks below were run or
reviewed on 2026-11-12 against main at
[`07b8b4e`](https://github.com/ikigai-rs/ikigai-llm/commit/07b8b4ede3cd15c1be8e9bc398d4f298334d0ad8).
The test command was `cargo test --all-targets`, using Cargo 1.99.0 and rustc 1.99.0.
[CI](.github/workflows/ci.yml) runs the shared Rust workflow for pull requests; the source
review and manual checks below are not automated by these tests.

| Check | Result | Evidence |
|---|---|---|
| Declared output faces and refusal of unsupported faces | Pass | `tests/as_faces.rs`, run by the command above. Checks served media types, defaults and refusal text; does not evaluate how a host presents the output. |
| Backend error bodies | Pass | `tests/error_bodies.rs`, run by the command above. Checks that HTML and plain-text error bodies are described rather than forwarded, and JSON reasons retained; does not establish that every reason is understandable. |
| Transport failure types | Pass | `tests/transport_errors.rs`, run by the command above against scratch servers. Checks failure classification and propagation; does not evaluate spoken error messages. |
| Redirect refusals | Pass | `tests/redirects.rs`, run by the command above. Checks status, location, missing location and refusal to follow redirects; does not test a person's ability to act on the advice. |
| Input refusals and sensitive text in errors | Pass | `tests/audit.rs`, run by the command above. Checks invalid typed inputs, capability boundaries, URL credential redaction and prompt removal from errors; not a general readability or security evaluation. |
| Streaming output and failures | Pass | `tests/streaming.rs`, run by the command above. Checks incremental completion delivery and stream failures; does not evaluate announcements, reading order or pacing in a host. |
| Meaning without color in crate-authored output | Pass | Source review of `src/lib.rs`, `src/streaming.rs` and `examples/ask.rs` on the main revision above: messages use words and add no color formatting. Model text and a host's styling are outside this check. |
| Plain-text completion truncation notice | Fail | Source review of the ask response in `src/lib.rs`: only the JSON envelope carries `finish_reason`; plain text returns the completion without a truncation notice. |
| Completion language metadata | Fail | Source review of `src/lib.rs` and `src/streaming.rs`: plain-text output carries no language metadata. A host must determine how to identify the model's language. |
| Actionable wording for every failure | Fail | Source review of `src/lib.rs`: `llm: no choices[0].message.content in response` and `llm: response was not JSON: …` name wire details without recovery advice. |
| Example error readability | Fail | Source review of `examples/ask.rs`: errors use debug formatting (`{e:?}`), rather than the error's sentence. No user comprehension study was run. |
| Documentation and output in a browser: headings, contrast, zoom and reflow | Blocked | No browser evaluation was run for this review. GitHub or another host renders the Markdown and output; no rendering environment was selected or exercised. |
| Manual keyboard | Blocked | No person performed a keyboard pass on the documentation or a host presenting this output; no interactive evaluation session was available. |
| Automated accessibility checker | Blocked | No axe or other accessibility checker was run; this crate has no browser interface or configured browser accessibility test. Rust tests do not substitute for this check. |
| Screen reader | Blocked | No screen reader was used on this output; no screen reader session was available for this review. |
| Screen magnifier | Blocked | No screen magnifier was used on this output; no magnifier session was available for this review. |
| Speech input | Blocked | No speech-input software was used on this output; no speech-input session was available for this review. |
| Switch access | Blocked | No switch-access software was used on this output; no switch-access session was available for this review. |
| Braille display | Blocked | No braille display was used on this output; no braille display was available for this review. |

No historical browser or assistive-technology run is offered as a current Pass.

## Known limitations

Confirmed by reading the source on 2026-11-12, as recorded in the table above:

- The `text/plain` answer does not say when a completion was cut short (for example at
  `max_tokens`); only the JSON envelope carries the finish reason.
- The `text/plain` answer states no language, and the completion may be in any language.
- Some failures name a wire detail rather than what to do: `llm: no choices[0].message.content
  in response`, and `llm: response was not JSON: …` followed by the parser's own text.
- `examples/ask.rs` prints an error in its debug form (`{e:?}`), which is harder to read than the
  error's sentence.
- This repository's own text is English only.

## Contributing

Read this file before changing any output listed above: the faces, the refusals or the example.
Keep meaning in words, and keep the model's text separate from this crate's. Say in the pull
request's accessibility section, give each check a `Check | Result | Evidence` row with
Pass, Fail or Blocked, the command or test and tool version, or why it could not run. Keep
unrun checks as Blocked, including assistive technology nobody used. A historical run that
CI does not repeat is Blocked with its date, not a current Pass.

## Reporting a barrier

[Open an issue](https://github.com/ikigai-rs/ikigai-llm/issues/new?labels=accessibility) with
the `accessibility` label (the link sets it when GitHub lets you add labels; otherwise a
maintainer adds it): what you were doing, what happened, and your terminal or other tool and any
assistive technology, if you are comfortable sharing them. Anything security-sensitive goes
through [private vulnerability reporting](https://github.com/ikigai-rs/ikigai-llm/security/advisories/new).
**We acknowledge a report within 7 days**, and promise nothing further.

## Change history

| Date | Change |
|---|---|
| 2026-10-31 | First version. |
| 2026-11-12 | Record scoped Pass, Fail and Blocked results, current main evidence and unrun assistive-technology checks. |
