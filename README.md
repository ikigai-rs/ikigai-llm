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
    "remote": { "base_url": "https://api.example.com/v1", "model": "gpt-4o", "api_key": "…" }
  }
}
```

```rust
let registry = ikigai_llm::Registry::from_json(&std::fs::read_to_string(path)?)?;
let space = ikigai_llm::space(Arc::new(my_transport), registry);
// urn:llm:ask -> fast · urn:llm:big:ask -> the 70B · source urn:llm:config -> the registry
```

`source urn:llm:config` shows the loaded registry with keys masked as `***`.

## Capability profiles & `urn:llm:models`
Each provider may declare a **`caps`** profile — `context` (tokens), `modalities`
(`["text","vision"]`), `tools`, `json`, `cost` (`local`|`cheap`|`premium`),
`params` (`"3B"`) — the traits selection will reason over. **`urn:llm:models`**
is the annotated inventory: JSON by default, and `as=text/turtle` renders the
**queryable trait graph** (`ik:LlmBackend` · `ik:model` · `ik:context` ·
`ik:modality` · `ik:tools` · `ik:cost` · `ik:vendor`), so "a vision model with
≥32k context" becomes a SPARQL query over a resource.

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
`tools` · `json` · **`vendor=x` / `vendor!=x`** (the governance axis — a provider
declares its `vendor` in caps, e.g. `ollama`/`openai`/`anthropic`, and
`vendor!=openai` means *this prompt never goes to OpenAI*) · `provider=name` /
`provider!=name` (registry entries by your local names).

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
