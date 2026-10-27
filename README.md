# ikigai-llm

Flexible LLM inference as ikigai ROC resources: one **facade grammar**
(`urn:llm:ask`) dispatches to pluggable **backend modules**, each also directly
addressable (`urn:llm:<provider>:ask`).

## What it binds today
- **`urn:llm:ask`** — the facade. Picks a backend (`provider=` arg, else the
  configured default) and re-issues the request to `urn:llm:<provider>:ask`
  through the kernel (so the backend's cache validity / golden threads propagate).
- **`urn:llm:<provider>:ask`** — one **OpenAI-compatible chat backend** over REST.
  That single shape covers **Ollama, vLLM, `llama.cpp`'s server, `mlx_lm.server`,
  and LM Studio** — they differ only in `base_url`, `default_model`, and whether a
  key is needed.

Buffered (no streaming yet), single-turn, `urn:cap:net`-gated. The gate judges
host, **port** and path exactly as `ikigai-http`'s own endpoints do: a grant on
`urn:cap:net:localhost:8000` reaches the MLX server on 8000 and not Ollama on
11434, and a deny on one port refuses only that port. Generation is
non-deterministic, so results are uncacheable by default.

Passes [`ikigai-conformance`](https://github.com/ikigai-rs/ikigai-conformance):
`tests/conformance.rs` walks every endpoint over a loopback stub speaking
Ollama's API, and pins by hand what the suite cannot see — every network action
refuses with a typed `Denied` before any socket opens (under no grants and under
a grant on another host), an answer is never served from cache, and every
config-derived result is cut by one golden thread, `urn:llm:config`. The space
is **host-named**: `space(transport, registry)` is built from what the host hands
it, one set of doors per configured provider, so it claims no
`urn:iki:space:llm` name and the host names the instance if it wants one.

### Inputs
`prompt` (or piped `content`) · `model` · `system` · `temperature` · `max_tokens`
· `as` (`application/json` for a `{text, model, usage}` envelope; default
`text/plain`) · `provider` and `needs` (facade only).

An optional input is either absent or used: one supplied in a form the endpoint
cannot read (a reference, interned content, non-UTF-8 bytes) is refused, never
treated as absent — on the facade, absent means "the default", and a `needs=`
policy it could not read must not route there. `temperature` must be a finite
number and `max_tokens` a positive integer, or the ask is refused before any
request is sent.

`as` is matched by the media type's essence against the faces each endpoint
declares (its `one_of`, which is what `urn:kernel:validate` checks):
parameters are ignored and case is folded, so `as=application/json;
charset=utf-8` is the JSON face, and a face the endpoint does not declare
(`application/ld+json`, `text/turtle` on an ask) is refused before any request
is sent, never answered with something else.

### Mounting it
The host injects an [`ikigai_http::HttpTransport`] (the same seam `ikigai-http`
uses — `ureq`/`reqwest` natively, `fetch` in the browser):

```rust
use std::sync::Arc;
use ikigai_llm::{space, OpenAiConfig};

let space = space(Arc::new(my_transport), OpenAiConfig::ollama("llama3.1"));
let kernel = ikigai_core::Kernel::new(Arc::new(space));
// urn:llm:ask  prompt="Explain ROC in one sentence"
```

## Provider registry & `urn:llm:config`
`space()` takes a **`Registry`** — several providers plus a default — and binds
one `urn:llm:<name>:ask` backend for each (all catalog-advertised), the
`urn:llm:ask` facade (routing to the default), and **`urn:llm:config`** (a
resource reporting the effective registry, **API keys redacted** and any
`user:password@` removed from a `base_url` — it answers under no capability at
all). A single `OpenAiConfig` still works via `From<OpenAiConfig>`.

The registry is compiled defaults ⊕ an optional hand-editable JSON file (the
load-time form of "the logical config aliases to file-or-code"):

```json
{
  "default": "fast",
  "providers": {
    "fast":   { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
                "caps": { "context": 131072, "modalities": ["text"], "tools": true,
                          "cost": "local", "params": "3B" } },
    "big":    { "base_url": "http://localhost:11434/v1", "model": "llama3.1:70b" },
    "rapid":  { "base_url": "http://localhost:8000/v1",
                "caps": { "cost": "local", "vendor": "mlx" } },
    "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "…" }
  }
}
```

An `api_key` goes out as `Authorization: Bearer <key>`, built by
`ikigai_http::auth::bearer`, so it must be an RFC 6750 `b64token`: letters,
digits and `-._~+/`, then any number of `=`. Every key the hosted providers
issue fits (`sk-proj-…`, `sk-ant-…`, `gsk_…`, `AIza…`). A key you choose for a
local vLLM or `llama.cpp` server must fit too: one with a space, a quote or a
line break is refused when the file loads, naming the provider and never the key.

### `model` is optional: name the server, not the model
An entry that **omits `model`** (like `rapid` above) names the **server**. The
model is discovered from the backend per resolve, so swapping the model behind
that server — a different `rapid-mlx` checkpoint, a freshly pulled Ollama tag —
takes effect on the next `ask`: no config edit, no host restart. That matters
because the registry is read **once at kernel construction**; there is no
watcher, so a pinned `model` costs a bounce of every host that read the file,
and the name lies in between.

Discovery reuses one existing rule rather than adding a second: the
smallest **chat-capable** model the server lists (see *Installed models* below),
so a big model stays an explicit choice. Several models served is a legitimate
state resolved by that rule, not an error — `rapid-mlx` lists its canonical id
*and* a lowercase alias for the same weights. Pin a `model` to say which.

Failure is loud and local: an unreachable backend errors with
`could not discover a model at <base_url>`, and **never** silently answers from
a different provider.

```rust
let registry = ikigai_llm::Registry::from_json(&std::fs::read_to_string(path)?)?;
let space = ikigai_llm::space(Arc::new(my_transport), registry);
// urn:llm:ask -> fast · urn:llm:big:ask -> the 70B · source urn:llm:config -> the registry
```

`source urn:llm:config` shows the loaded registry with keys masked as `***` and
URL credentials removed (`https://user:pass@proxy/v1` reads `https://proxy/v1`);
`urn:llm:models` and every error that names a `base_url` show the same redacted
form. Those are the only secret-bearing fields an entry has. The key is sent on
**every** request to its provider — chat, the model listing, liveness, and
Ollama's native `/api/tags` and `/api/show` — so a keyed provider is not reported
down, and a keyed provider that names only its server can discover its model.

## Capability profiles & `urn:llm:models`
Each provider may declare a **`caps`** profile — `context` (tokens), `modalities`
(`["text","vision"]`), `tools`, `json`, `cost` (`local`|`cheap`|`premium`),
`batchAt` (load shape, see below), `params` (`"3B"`) — the traits selection
reasons over.

**Two axes cut `caps`, and they are not the same axis.** *Use*: selection
**routes** on `context`, `modalities`, `tools`, `json`, `cost`, `vendor` and
`batchAt` — a wrong value misroutes work; only `params` is display.
*Provenance*: who is a trustworthy witness. `context`, `modalities` and `tools`
are the **server's** to know — for a provider that names only the server they
are read from that server's listing, because a hand-written value that survives
a model swap silently misroutes work. `cost`, `vendor` and `batchAt` are
**declared-only**: never discovered, because they are exactly the axes a policy
excludes on. A server that self-reports `owned_by: "rapid-mlx"` must not be able
to launder itself past `vendor!=openai` by saying so — so the discovered profile
has no field to put it in, and a provider that declared no vendor still fails the
exclusion. (This README used to call that group *governance* as if it meant "not
routed on"; it never did. The word names the provenance, not the use.)

Declared values always win; discovery fills only gaps. `modalities` is a set,
so there filling gaps is a union: discovery adds a modality the declaration did
not list, and a declaration or annotation adds one the server under-reports.
**`urn:llm:models`**
is the annotated inventory: JSON by default, and `as=text/turtle` renders the
**queryable trait graph** (`ik:LlmBackend` · `ik:model` · `ik:context` ·
`ik:modality` · `ik:tools` · `ik:cost` · `ik:vendor` · `ik:batchAt`), so "a
vision model with ≥32k context" becomes a SPARQL query over a resource.
Every one of those terms is the **shared** vocabulary's, published at
<https://ikigai-rs.dev/ns> — so the graph federates with every other ikigai
graph instead of being module-local jargon, and the conformance walk fails on
any term the vocabulary does not define. Literals carry the datatype the
vocabulary's `rdfs:range` declares, which is what makes the Turtle and JSON-LD
faces of the same inventory diff clean.

Trait facts arrive at three strengths — **annotations > declared > discovered**:

- **Discovered** (weakest, gaps only): a provider that declares `vendor:
  "ollama"` opts into live discovery via Ollama's native `/api/show` — context
  length, vision/tools capabilities, parameter size — merged declared-wins.
  Graceful: server down or capability missing → the declared profile stands.
  (The vendor declaration is the opt-in; unknown vendors are never probed with
  your model names.)
- **Declared**: the config file's `caps` block.
- **Annotations** (strongest): `Registry::apply_annotations(facts)` takes triples
  from an alignment/annotation graph (subjects are the trait-graph's own
  `urn:llm:<name>:ask` IRIs, or bare provider names) and **completes or corrects**
  under-specified descriptions — an override is never silent: every conflict is
  returned for the host to log. `modality` facts union in, and keep whatever
  discovery finds beside them. So a config that
  forgot `vendor` on a remote can be fixed from the graph, and `vendor!=openai`
  then correctly excludes it instead of conservatively failing everything.

## Capability-based selection: `urn:llm:select` & `needs=`
Stop naming models — state requirements. **`urn:llm:select needs="…"`** resolves a
requirement expression over the declared trait profiles and returns the winning
backend IRI; the **facade accepts the same `needs=`** and routes the ask directly:

```text
source urn:llm:select needs="vision, ctx>=32k, cost<=cheap"     -> urn:llm:seer:ask
source urn:llm:ask needs="ctx>=100k" prompt="…"                 -> asks the winner
```

Grammar (comma-separated): `ctx>=N` (or `Nk` = ×1024) · `cost<=tier` / `cost=tier`
(`local` < `cheap` < `premium`) · `modality=x` or bare `text`/`vision`/`audio` ·
`tools` · `json` · **`vendor=x` / `vendor!=x`** (a provider declares its `vendor`
in caps, e.g. `ollama`/`openai`/`anthropic`, and `vendor!=openai` means *this
prompt never goes to OpenAI*) · `provider=name` / `provider!=name` (registry
entries by your local names) · **`batchAt<=N`** (load shape — see the next
section).

Selection reasons only over the backends **the caller's capability can reach**:
a backend you could not ask is not an answer, and naming one would hand you the
identity of a backend your grant withholds. So `urn:llm:select` declares
`urn:cap:net:*` (with no net grant it can never answer), the facade's `needs=`
routes to a reachable backend, and when only unreachable backends satisfy the
requirement the refusal is a typed `Denied` that names none of them.

Unknown terms **error** (a typo must not mis-select); a trait a provider didn't
declare can't satisfy a requirement on it — **including `vendor!=`**: an
undeclared vendor fails the exclusion, because it might *be* that vendor. Policy
among matches: **cheapest-that-fits → smallest context → registry order**.
Routing precedence on the facade: `provider=` → `needs=` → the configured default.

```text
source urn:llm:ask needs="ctx>=32k, vendor!=openai" prompt="…"   # governance-constrained ask
```

Selection is deterministic plain code over the registry — the SPARQL power path
is *composition*, not a dependency: `urn:llm:models as=text/turtle` is the same
trait data as a queryable graph.

## Load shape: `batchAt` and fanning out
Local backends differ most on an axis no trait expressed until now:
**throughput versus latency**. Measured on one machine, same model (Qwen3 27B)
either side, aggregate tok/s:

| concurrent | 1 | 2 | 3 | 8 | 10 |
|---|---|---|---|---|---|
| a batching server (continuous batching) | 28.9 | 40.1 | **46.7** | **79.8** | **65.9** |
| a serializing server | **53.0** | 41.0 | 35.7 | 42.1 | 40.5 |

One request: the serializing backend is ~1.8× better. Ten: the batching one
finishes the work in 17.9s against 49.3s. Without a trait for it, a caller that
knows it is fanning out has no way to say so — it names a provider, and bakes one
machine's topology into its call site.

`batchAt: N` declares **the concurrency at or above which this backend is the
throughput winner** — the operator's own measured crossover, in requests:

```json
"rapid": { "base_url": "http://localhost:8000/v1",
           "caps": { "cost": "local", "vendor": "mlx", "batchAt": 2 } }
```

A caller then states its **operating point**, not a provider:

```text
source urn:llm:select needs="batchAt<=10"          -> the backend that batches
source urn:llm:ask needs="batchAt<=10" prompt="…"  # one leg of a 10-way fan-out
```

`batchAt<=10` reads exactly the way `cost<=cheap` does — an upper bound on the
*declared* value — and means "my fan-out is 10 wide; the backend's crossover must
be at or below that."

**There is no default, deliberately.** The crossover above landed at 2, but that
is one machine, one model, one prompt length and one quantization pair; it is a
measurement, not a constant, and nothing in the wire protocol can correct a wrong
guess — no OpenAI-compatible endpoint advertises whether it batches. So:

- an **undeclared** provider is *unknown*, and unknown satisfies no `batchAt<=`
  requirement (the `vendor!=` rule: silence is not a claim to batch);
- a **server** cannot assert one — the discovered profile has no field for it;
- an **annotation graph** can (`ik:batchAt`), because it is operator-authored;
- `batchAt: 0` and a non-numeric value **fail the config load**, naming the
  provider, rather than defaulting to absent — as do a mistyped key
  (`batchat`), a `cost` outside the three tiers, and a `default` that names no
  configured provider.

Omitting the term routes exactly as it did before the trait existed.

## Installed models & the default-model fallback
**`urn:llm:<provider>:installed`** lists what the provider can actually serve
right now, **smallest-first** — a declared `vendor: "ollama"` uses the native
`/api/tags` (which reports sizes; the `as=application/json` face carries them
for host co-load budgeting), anything else falls back to the OpenAI-compat
`GET {base}/models` (names only, server order). Newline list, pipeable. It's
the complement of `urn:llm:models`: *configured* vs *installed*. Live fact —
uncacheable. Smallest-first means "first installed" reads as "cheapest to
run": big models are an explicit choice (`model=` / `needs=`), never an
accident of list order.

And the backend resolves defaults against it: if a request **didn't name**
`model=` and the configured default 404s (the demo moved machines; the model
was never pulled), it lists what's installed and **retries once with the
smallest model the server says can chat** — Ollama's `/api/show` reports
`completion`. A substitution the caller did not ask for needs that evidence: a
listing that says nothing about capabilities (the OpenAI-compat `/models`) could
put an embedder first, so there the 404 surfaces instead. An explicit `model=`
is *never* substituted — that errors honestly. So an Ollama host's default
config degrades to "use what's here" instead of failing on a hardcoded name.

## Model identity: `urn:llm:<provider>:model`
The model id serving this provider, as `text/plain` — e.g.
`source urn:llm:coder:model` → `qwen3-coder:30b`. The cheap identity face for
consumers that fold true model identity into derived artifacts (archive
version tags, provenance labels) without pulling the whole `urn:llm:config`
registry JSON. Model ids aren't secrets — nothing is redacted.

**The provider's own config picks the cost contract**, so a consumer's cost is
whatever its providers chose:

| provider | answer | network | capability | cacheable |
| --- | --- | --- | --- | --- |
| pins a `model` | that id, **verbatim** | none | none | yes (`Never`) |
| names the server | the id the server serves **now** | one `GET {base}/models` | `urn:cap:net:*` | **no** |

A pinned provider is a pure config read, exactly as before — that matters
because `ikigai-browse` keys explain-archive version tags on this resource, and
a changed id silently re-derives every archived explanation. A discovering
provider has no configured default, so the discovered id is the only honest
answer, and it re-keys the archive exactly when the model behind the server
really changes — which is the behaviour browse already documents as a feature.
It is uncacheable on purpose: caching a discovered id restores the staleness
discovery exists to remove, and a cached representation reached through a mount
can never be invalidated.

`urn:llm:models` and `urn:llm:select` follow one rule: **cacheable exactly when
answering asked no server anything.** A provider that names only its server
asks for its model list, and a declared `vendor: "ollama"` opts into a live
`/api/show` on every resolve, so either makes the answer a live fact — even with
every model pinned. Whether a server was asked is recorded at the transport, and
a probe the caller's capability could not make leaves the answer a function of
config and capability, cacheable under `urn:llm:config`.

## Liveness: `urn:llm:<provider>:up`
A boolean resource — `true` if the provider answers a cheap `GET {base_url}/models`,
else `false`. Built for `urn:fn:conditional`, so demos degrade gracefully:

```text
source urn:fn:conditional if=urn:llm:ollama:up then=urn:data:jury else=urn:data:ollama-offline
```

Uncacheable (liveness is a live fact); a capability that can't reach the host is
an error, not `false` (denied ≠ down).

## Failures are typed, and each type says who may retry it

Every failure to get an answer is typed by `ikigai-http`'s one table
(`ikigai_http::status`, since 0.16.0; ledger #1546), and the variant's
**class** (`Error::class`, ikigai-core 0.1.108) says who may repeat it.
ikigai-throttle's Retry, CircuitBreaker and Failover act on that class in front
of `urn:llm:*`:

| What happened | Error | Class |
|---|---|---|
| connection refused, DNS failure, connection dropped | `Unavailable` | transient |
| the request timed out, or the backend answered 408 or 504 | `Timeout` | transient |
| the backend answered 502, or 503 with no usable `Retry-After` | `Unavailable` | transient |
| the backend answered 429, or 503 with `Retry-After` | `RetryAfter`, with the delay | retry-after |
| the backend answered 413, 414 or 431 | `OverBudget` | over-budget |
| the backend answered 401 or 403 (it refused the configured key) | `Refused` | refused |
| the backend answered 404 or 410 | `NotFound` | permanent |
| the backend answered 409 or 412 | `Conflict` | permanent |
| the backend answered 501 | `Unsupported` | permanent |
| the backend answered 500 or another 5xx, any other 4xx, or a 2xx that is not a completion | `Endpoint` | unknown |
| no capability to reach the host | `Denied` | denied |

A `Retry-After` is read in both of its forms: delay-seconds, and an HTTP-date
measured on the kernel's clock. Every action declares the classes it can fail
with (`Description::fails`), and `tests/conformance.rs` holds it to them
(conformance's FAILURES check), with two probes against a fake origin: each ask
fails as `retry-after` under a 429 and as `refused` under a 401.

The type survives the `urn:llm:ask` facade and the wrapping a discovering
provider adds (`could not discover a model at …`), a `RetryAfter`'s delay
included. A transport reports failure as text, so a timeout is recognized by
its words (`timed out`, `timeout`) with the URL removed; anything else
unanswered is `Unavailable`. The message is this crate's, with any
`user:password@` in the URL scrubbed. Pinned by `tests/transport_errors.rs`
against scratch servers on 127.0.0.1.

## 0.17.0 (unreleased)

Minor, **BREAKING for hosts** (the core 0.2.0 sweep, ledger #2261).

- **BREAKING: `ikigai-core` moves to 0.2.0**, the streams release. Core's
  types are in this crate's API, and the old `0.1.108` pin meant `<0.2.0`, so a
  host moves to core 0.2 with this release or links two cores.
- **BREAKING for hosts: `ikigai-http` moves to 0.4.0**, its release on core
  0.2.0. A transport's `HttpResponse::body` is now an `HttpBody`, a stream:
  return a canned body with `bytes.into()`, and wrap a real client's reader
  with `HttpBody::from_reader` instead of reading it first (the `ask` example
  shows ureq's, through `futures::io::AllowStdIo`).
- **A provider's answer is read to at most 8 MiB** (ikigai-http's
  `DEFAULT_MAX_RESPONSE_BYTES`). A larger answer is `OverBudget`; an error
  answer whose body is larger keeps its status's type, the body described as
  unread. `urn:llm:<provider>:up` reads no body at all.
- `ikigai-conformance` moves to 0.11.0, whose DATATYPES and FAILURES checks
  run by default; the `ik:batchAt` graph is now held to DATATYPES as well. No
  finding: the literal was already tagged `^^xsd:positiveInteger`.

## 0.16.0 (2026-10-18)

Minor, **BREAKING for hosts and for callers matching error variants** (ledger
#1546).

- **BREAKING for hosts: `ikigai-http` moves to 0.3.0.** `space(transport,
  registry)` takes an `Arc<dyn ikigai_http::HttpTransport>`, and a host shares
  one transport between `ikigai-http`'s space and this one, so both must name
  the same `ikigai-http`: move it to 0.3 with this release. The `ikigai-core`
  floor moves 0.1.86 → 0.1.108.
- **Every failure is typed by `ikigai-http`'s one status table**
  (`ikigai_http::status`). This crate's own copy is deleted; it had drifted. A
  caller matching an old variant now sees a new one:
  - an origin's **401/403** is `Refused` (was `Endpoint`), **404** is
    `NotFound` (was `Endpoint`), 409/412 `Conflict`, 413/414/431 `OverBudget`,
    501 `Unsupported` (all were `Endpoint` or `Unavailable`);
  - **429** is `RetryAfter`, carrying the `Retry-After` delay when the header
    parses (was `Unavailable`). It is **not transient**: an overlay built on
    `is_transient` no longer re-issues straight into the limit, and one reading
    `Error::class` waits out the delay when its deadline allows. A 503 with
    `Retry-After` is `RetryAfter` too;
  - **500** and the other 5xx but 502, 503 and 504 are `Endpoint`, class
    unknown (were `Unavailable`): they are **no longer retried**. 502, a 503
    without a usable delay, 408/504 and a request that got no answer are
    unchanged.
- **Rewording keeps the type.** A discovering provider's
  `could not discover a model at …` now keeps `Refused`, `RetryAfter` (with its
  delay), `OverBudget`, `NotFound`, `Conflict` and `Unsupported`; `NotFound`
  and `Conflict` used to become `Endpoint`.
- **Every action declares its failure classes** (`Description::fails`), so the
  manifold and an agent's retry plan read them from the contract. The
  conformance test (ikigai-conformance 0.10.0) holds each action to them, and
  probes each ask against a fake origin's 429 and 401.

## 0.15.1 (2026-10-18)

Patch: no public API changes, and the dependency floor stays inside the same
ranges.

- **The `Authorization` header is built by `ikigai_http::auth::bearer`**
  (ledger #1874), the ecosystem's one RFC 6750 builder, rather than by hand.
  The `ikigai-http` floor moves 0.2.0 → 0.2.1, where that builder first ships;
  same 0.2 range, so a host sharing one transport is not moved.
- **An `api_key` that is not a `b64token` is refused**: at
  `Registry::from_json`, and on every request for an `OpenAiConfig` built in
  code, before anything reaches the transport. The error names the provider
  and the rule code (`bearer-token-syntax`), never the key. Before, a CR/LF in
  a key went into the header value unchecked. Every hosted provider's issued
  key fits. A key an operator chose for a local vLLM or `llama.cpp` server
  with a space or a quote in it is now refused, and has to be changed.

## 0.15.0 (2026-10-17)

- **BREAKING for hosts: `ikigai-http` moves to 0.2.0** (the naming wave,
  ledger #11). Nothing in this crate's behavior changes, and it names no http
  endpoint. But `space(transport, registry)` takes an
  `Arc<dyn ikigai_http::HttpTransport>`, and a host normally shares ONE
  transport between `ikigai-http`'s space and this one, so both must name the
  same `ikigai-http`: a host still on 0.1.x gets a trait from a different crate
  version and fails to compile. Move `ikigai-http` to 0.2 with this release.
  There, `ikigai_http::space()` returns core's `Alias` overlay rather than an
  `EndpointSpace` (hold it as `Arc<dyn Space>`), the endpoints are
  `urn:http:get` … `urn:http:delete` with ids `http-get` … `http-delete`, and
  `urn:httpGet` … `urn:httpDelete` resolve as deprecated aliases until
  `ikigai-http` 0.3.0. See its CHANGELOG.
- **The `ikigai-core` floor moves 0.1.48 → 0.1.86**, `ikigai-http` 0.2.0's own
  core floor. A fresh resolve of `src/` alone at core `=0.1.86` and http
  `=0.2.0` compiles.

## 0.14.0 (2026-10-13)

Minor: two changes alter which error variant a caller sees.

- **Failures are typed for the reliability overlays** (ledger #88). Every
  failure was `Error::Endpoint`, so ikigai-throttle's Retry, CircuitBreaker
  and Failover never engaged. A refused connection, a DNS failure and a 429/5xx
  answer are now `Unavailable`; a timeout and a 408/504 answer are `Timeout`;
  any other 4xx and a malformed answer stay `Endpoint`. A discovering
  provider's `could not discover a model` wrap keeps the inner type. See
  [Failures are typed](#failures-are-typed-for-the-reliability-overlays);
  pinned by `tests/transport_errors.rs`, 8 of whose 11 tests failed on 0.13.1.
- **`as=` matches the declared faces** (ledger #179): by the media type's
  essence (parameters ignored, case folded), against the same list the
  manifold's `one_of` carries. An undeclared face is refused with
  `InvalidArgument` before any request is sent; it used to be answered
  (`application/ld+json` got JSON) or silently given the default. Pinned by
  `tests/as_faces.rs`.
- **`rust-version = "1.88"`**, the measured floor (ledger #181), so the shared
  CI's msrv job runs clippy and the tests on it instead of skipping.

## 0.13.1 (2026-10-10)

- **An error body this module cannot read as a structured error is described,
  not forwarded** (ledger #178). An nginx error page or a proxy's plain-text
  refusal used to go into the error message whole; it now contributes
  `unstructured error body (text/html, 162 bytes)` (the `Content-Type`
  header's `type/subtype`, or `no content type`, and the length), and an empty
  body says `empty error body`. What still passes through is unchanged: the
  status code, and a JSON body's stated reason with the prompt cut from it.
  Pinned by `tests/error_bodies.rs`, which failed on 0.13.0.

## 0.13.0 (2026-10-08)

Fixes from the unled audit of 0.12.2 (ledger #884), each pinned by a test in
`tests/audit.rs` that failed on 0.12.2:

- **Port-scoped net grants are enforced.** Every gate called the port-less
  `ikigai_http::net_allows`, which treats the port as unknown, so
  `urn:cap:net:localhost:8000` reached Ollama on 11434 and a deny on one port
  refused every port. They now call `net_allows_port` with the URL's real (or
  scheme-default) port. The `ikigai-http` floor moves to 0.1.7, the first
  release with `net_allows_port`.
- **URL credentials are redacted** from `urn:llm:config`, `urn:llm:models` and
  every composed error (a transport's own error text included).
- **The API key is sent on every request** to its provider, not only on chat.
- **An inventory or selection built from a live probe is no longer cached for
  ever.** A pinned `vendor: "ollama"` provider still probes `/api/show` on every
  resolve, yet `urn:llm:models` and `urn:llm:select` were marked permanent, so a
  selection taken while Ollama was down was served until restart. They are now
  uncacheable whenever building them asked a server. The cost, measured on a
  three-provider registry with two `vendor: "ollama"` entries (stub transport,
  release build): a repeat `urn:llm:select` read went from a ~0.6µs cache hit to
  ~6.7µs with a zero-latency transport and ~2.5ms with a 1ms `/api/show` — one
  round trip per declared-ollama provider, which the facade's `needs=` already
  paid on every ask. A registry that opts into no discovery caches as before.
- **`urn:llm:select` answers only with a backend the caller can reach**, and
  declares `urn:cap:net:*`; a caller holding no net grant is refused by the
  kernel's floor instead of being told about backends it cannot use.
  `urn:llm:models` still needs no capability: it probes only hosts the caller's
  grant reaches and otherwise reports the declared profile.
- **The config refuses what it used to load and misroute on**: an unknown key at
  any level (`Caps` now denies unknown fields wherever it is deserialized), a
  `cost` outside `local` | `cheap` | `premium`, and a `default` that names no
  provider.
- **Inputs are refused rather than dropped**: an unreadable `needs=` or
  `provider=` on the facade (it used to route to the default — the very vendor a
  `vendor!=` policy excluded), an unreadable `model`, `system` or `supports`, a
  `temperature` that is not a finite number (NaN was sent as JSON `null`), and a
  `max_tokens` that is not a positive integer.
- `ctx>=Nk` beyond 64 bits is a grammar error; it panicked in debug builds and
  wrapped in release, selecting a 128k backend for a requirement nothing meets.
- The Turtle face escapes LF and CR, so a server-reported model id or modality
  with a line break no longer makes the trait graph unparseable.
- The trace labels the model that **answered**, not the configured one that
  404'd before the fallback.
- An error body contributes only its stated reason (`error.message`, a
  `message`/`detail` string, FastAPI's `detail[].msg`), or its field names when
  it has none of those; the prompt and system prompt are cut from it wherever a
  server echoed them. FastAPI's 422 used to put the prompt in the error.
- An annotated `modality` keeps the discovered ones beside it.
- The 404 default-model fallback substitutes only a model the server says can
  chat; on the OpenAI-compat path, which says nothing, it no longer retries
  with whatever is listed first.

**Version: 0.13.0 (minor), not 0.12.3.** A registry that loaded under 0.12.2 can
now be refused (the three load checks above, and `Caps` deserialization
generally), requests 0.12.2 answered are now refused, and `urn:llm:select`
refuses a caller with no net grant. Under 0.x caret rules a patch reaches every
`^0.12` consumer on a plain `cargo update`, and ikigai-cli falls back to its
built-in single-Ollama registry when `llm.json` fails to parse — so a config
refusal arriving unasked would silently reroute a host. A minor makes each
consumer (ikigai-cli and ikigai-dev-server pin `0.12.1`, ikigai-cms-web `0.12`)
opt in with a manifest edit, which is the moment to check its `llm.json` loads.
The cost is that the security fixes reach no consumer until that edit.

## Design & roadmap
The facade is the imperative seed of the interception/rewrite primitive: a static
alias would be a `Rewrite` space, but selection that reads args/config is an
endpoint whose `invoke` does the rewrite. Deferred: **value subsumption in
selection** (`ik:AzureOpenAI ⊑ ik:OpenAI` so `vendor!=openai` closes over the
hierarchy; `ik:Vision ⊑ ik:Multimodal`), **live-reload** (make `urn:llm:config` a
live resource that sources the file under a golden thread), **transrepted config**
(author YAML/Turtle, transrept through the kernel), `key_ref` → the secrets infra,
deterministic caching, json mode/tools, in-process `llama.cpp` (FFI) + MLX (pyo3),
streaming, and `urn:llm:embed`.
