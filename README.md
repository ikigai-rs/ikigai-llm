# ikigai-llm

Flexible LLM inference as ikigai ROC resources: one **facade grammar**
(`urn:llm:ask`) dispatches to pluggable **backend modules**, each also directly
addressable (`urn:llm:<provider>:ask`).

## Slice 0 (this crate today)
- **`urn:llm:ask`** — the facade. Picks a backend (`provider=` arg, else the
  configured default) and re-issues the request to `urn:llm:<provider>:ask`
  through the kernel (so the backend's cache validity / golden threads propagate).
- **`urn:llm:<provider>:ask`** — one **OpenAI-compatible chat backend** over REST.
  That single shape covers **Ollama, vLLM, `llama.cpp`'s server, `mlx_lm.server`,
  and LM Studio** — they differ only in `base_url`, `default_model`, and whether a
  key is needed.

Buffered (no streaming yet), single-turn, `urn:cap:net`-gated. Generation is
non-deterministic, so results are uncacheable by default.

### Inputs
`prompt` (or piped `content`) · `model` · `system` · `temperature` · `max_tokens`
· `as` (`application/json` for a `{text, model, usage}` envelope; default
`text/plain`) · `provider` (facade only).

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
resource reporting the effective registry, **API keys redacted**). A single
`OpenAiConfig` still works via `From<OpenAiConfig>`.

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

`source urn:llm:config` shows the loaded registry with keys masked as `***`.

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

Declared values always win; discovery fills only gaps. **`urn:llm:models`**
is the annotated inventory: JSON by default, and `as=text/turtle` renders the
**queryable trait graph** (`ik:LlmBackend` · `ik:model` · `ik:context` ·
`ik:modality` · `ik:tools` · `ik:cost` · `ik:vendor` · `ik:batchAt`), so "a
vision model with ≥32k context" becomes a SPARQL query over a resource.

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
  returned for the host to log. `modality` facts union in. So a config that
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
  provider, rather than defaulting to absent.

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
was never pulled), it lists what's installed and **retries once with the first
available model**. An explicit `model=` is *never* substituted — that errors
honestly. So a host's default config degrades to "use what's here" instead of
failing on a hardcoded name.

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

The same rule governs `urn:llm:models` and `urn:llm:select`: cacheable while
every provider is pinned, uncacheable once any provider discovers.

## Liveness: `urn:llm:<provider>:up`
A boolean resource — `true` if the provider answers a cheap `GET {base_url}/models`,
else `false`. Built for `urn:fn:conditional`, so demos degrade gracefully:

```text
source urn:fn:conditional if=urn:llm:ollama:up then=urn:data:jury else=urn:data:ollama-offline
```

Uncacheable (liveness is a live fact); a capability that can't reach the host is
an error, not `false` (denied ≠ down).

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
