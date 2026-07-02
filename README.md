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
    "fast":   { "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b" },
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

## Design & roadmap
The facade is the imperative seed of the interception/rewrite primitive: a static
alias would be a `Rewrite` space, but selection that reads args/config is an
endpoint whose `invoke` does the rewrite. Deferred: **live-reload** (make
`urn:llm:config` a live resource that sources the file under a golden thread),
**transrepted config** (author YAML/Turtle, transrept through the kernel),
`key_ref` → the secrets infra, native Ollama backend + `urn:llm:models`,
deterministic caching, json mode/tools, in-process `llama.cpp` (FFI) + MLX
(pyo3), streaming, and `urn:llm:embed`.
