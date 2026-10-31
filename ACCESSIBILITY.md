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
| Last reviewed | 2026-10-31 |

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
  discovered, and a backend's failure typed by status.
- **`examples/ask.rs`**, which prints the model, the prompt, and the answer or the error.

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
- A backend's error page or plain-text refusal is described (status, media type, length), never
  forwarded, and a URL's `user:password@` is removed from error text.
- No color is written; meaning is in words.

## How it is tested

- `tests/as_faces.rs`: `an_undeclared_face_is_refused`, `every_declared_face_is_served_and_absent_means_the_default`.
- `tests/error_bodies.rs`: `an_html_error_page_is_described_not_forwarded`,
  `a_plain_text_refusal_is_described_not_forwarded`, `a_json_error_keeps_its_reason`.
- `tests/transport_errors.rs` pins the type of each failure against scratch servers.
- These pin what reaches the caller, not how it reads. **No automated accessibility checker,
  browser or assistive technology has been used on this output.**

## Known limitations

Found by reading the source on 2026-10-31:

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
request what was tested, with what, and what was not.

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
