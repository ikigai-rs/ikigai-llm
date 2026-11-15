# Budgets at the LLM door

Design spike for ledger #3752. Status: proposed; no budget enforcement ships in
this change. The organizational-grants graph in the item is the design input,
not a verified description of another repository. All behavior below is proposed
unless explicitly labeled as existing. Decisions for Brian appear at the end.

## Problem and reproduced baseline

A team must be able to give agents parts of its LLM allowance without creating
more allowance. Possessing a network grant must not imply unlimited inference.
The door must reserve enough quantity before sending a billable request, then
settle its actual cost. A counter checked and updated in separate operations
would let two requests spend the last unit.

Existing code read for this spike:

| Fact | Evidence |
|---|---|
| The facade copies arguments and reissues through the kernel; direct provider asks are also endpoints. | [src/lib.rs:701–736](../src/lib.rs#L701), [src/lib.rs:778–835](../src/lib.rs#L778) |
| Ask declares the network scope; providers with secret-backed keys add their use scope, and the exact scope is checked before reading the key. | [src/lib.rs:1192–1244](../src/lib.rs#L1192), [src/lib.rs:831–840](../src/lib.rs#L831), [src/lib.rs:2899–2918](../src/lib.rs#L2899) |
| `max_tokens` is optional and forwarded as an output limit; it is not an account balance. | [src/lib.rs:865–899](../src/lib.rs#L865) |
| A default-model 404 can cause a second completion attempt with a different model. | [src/lib.rs:909–937](../src/lib.rs#L909) |
| Buffered JSON exposes usage or null; plain text exposes only completion text. Generation representations have no explicit cache validity. | [src/lib.rs:969–1000](../src/lib.rs#L969) |
| The stream parser discards usage-only events and ends at `[DONE]`; dropping the body drops the reader. | [src/streaming.rs:23–24](../src/streaming.rs#L23), [src/streaming.rs:80–98](../src/streaming.rs#L80) |
| The library injects its transport; its declared core dependency is 0.3.0. | [src/lib.rs:779–796](../src/lib.rs#L779), [Cargo.toml:33–54](../Cargo.toml#L33) |
| Embedding has its own routed path, so an ask-only gate would not cover all inference. | [src/embed.rs:50–95](../src/embed.rs#L50) |

Reproduction: `cargo test --test budget_baseline` exercises the public kernel
API against a counting mock transport. A network-only capability reaches both
the facade and direct backend, making two successful calls whose responses omit
usage. This pins a missing prerequisite rather than claiming a regression: the
spike asks for a new policy. No production code changes are needed to reproduce
it. The test should be revised when mandatory budgets are implemented.

## Authority and quantity

A budget grant is a reference to an authoritative allocation, not a mutable
number copied into a bearer token. Proposed capability data binds a principal,
assignment ID and revision, budget ID, unit, maximum allocation, provider/model
and action scope, validity interval, period policy, and parent assignment.
Copying a capability shares the same allocation and meter; it never copies a
balance. Request arguments can select an authorized budget, never supply its
quantity, principal, price, clock, or ancestry. If several grants apply, require
an explicit selection; do not drain arbitrary budgets or silently combine them.

Proposed scope family: `urn:cap:llm:budget:spend:{id}`. The attached constraint
must survive attenuation, reissue, mounts, and serialization. A child may narrow
scope, validity and quantity, never widen them. Administrative authority does
not implicitly authorize spending, and a wildcard cannot mint quantity. Host
root configures allocations through the same transactional service; metered
inference has no implicit root exemption.

This repository's source above establishes scope checks, not a structured
quantity protocol in core. Do not infer that core can enforce numeric attenuation
from string matching. Decision 1 requests a core contract review and, if needed,
an opaque budget-constraint extension that preserves assignment identity and
attenuation. Core need not own financial counters: a host-injected authoritative
budget service owns them. A compatibility prototype could carry only a scoped
allocation reference, with all constraints checked by that service. It must not
encode an independently spendable amount in a scope string. No core changes are
part of this spike.

## Resource map

These proposed names describe sets with singular, type-first names and explicit
keyed partitions. IDs are opaque, canonical escaped segments. Budget names in
refusals are these IDs, not free-form labels containing private information.
The service is logically one authority even when multiple LLM doors use it.

| Resource set and keyed member | Kind and state |
|---|---|
| `urn:llm:budget`, `urn:llm:budget:id:{id}` | Budget atom: principal, parent, unit, allocation schedule, scope, validity, owner, reason, approver, assignment and policy revision. One unit per budget. |
| `urn:llm:meter`, `urn:llm:meter:budget:{id}:period:{start}` | Meter atom: immutable period bounds and allocation, committed child allocations, settled spend, outstanding reservations, revision and frozen flag. |
| `urn:llm:reservation`, `urn:llm:reservation:budget:{id}:period:{start}:id:{nonce}` | Internal durable atom: held quantities, attempt state, pinned price/tokenizer/model, settlement and outbox status. Nonce is random and is not a person identifier. |
| `urn:llm:proposal`, `urn:llm:proposal:id:{id}` | Proposal atom: base revision, exact diff, requester and approval identities; applied once or rejected. |
| `urn:llm:budget:id:{id}:balance` | Uncacheable composition of budget policy and current meter; not a separately writable balance. |
| `urn:llm:budget:id:{id}:child` | Authorized composition listing direct child budget references; not an organization-wide directory. |
| `urn:llm:ask`, `urn:llm:{provider}:ask` | Compositions of selection, authority checks, reservation, provider execution and settlement; neither holds the authoritative balance. |

The organization tree, roles, assignments and kinds-of-work taxonomy remain
upstream inputs. A role template alone grants no spend. Its approved assignment
resolves to budget atoms. This module checks the resulting action/provider/model
scope, not ledger labels or a new local copy of the organization graph.

## Actions, declarations and enforcement

The following are proposed semantic actions; the implementation must map them to
supported kernel verbs in its API review. Do not invent a verb in a published
contract merely because this table names an action. Every action describes its
inputs, scope requirements and failure classes in its endpoint description.
Parameterized wildcard declarations are discovery floors, followed by exact
resource checks at execution, as with the existing network pattern above.

| Action | Declared capability and execution check |
|---|---|
| Ask, both facade and direct backend | `urn:cap:net:*`, `urn:cap:llm:budget:spend:*`, and existing provider-use scopes when applicable. Backend enforces the exact host/port, provider key scope, principal, live allocation, scope and sufficient reservation. Facade passes authority without charging twice. |
| Read budget, balance or children | `urn:cap:llm:budget:read:*`; enforce exact budget scope and filter child listings. Spend does not grant access to other principals' allocations. |
| Propose create, split, narrow, revoke or widen | `urn:cap:llm:budget:propose:*`; check exact parent or budget authority and base revision. Every edit has a proposal diff, including reductions. |
| Approve and apply proposal | `urn:cap:llm:budget:approve:*`; exact scope, live revision and transactional invariants. A widening requires a different authenticated person from the requester, including root-initiated changes. |
| Reserve, settle and reconcile | Internal `urn:cap:llm:meter:write:*`, held by the door service only. Enforce exact allocation and reservation state. A caller's spend grant cannot edit counters or supply actual usage. |
| Deliver settlement log | Internal log-write authority limited to the settlement destination; no caller can append settlement facts. |

Check permission before revealing existence. An unauthorized lookup is Denied
regardless of whether the ID exists. Budget checks also apply to direct provider
entry points and any retry; wrapping only the facade is insufficient. Recommend
including embedding misses in the same budget policy before making an
organization-wide enforcement claim. A separate embedding rollout must be
explicit and must not offer an alternate path around an ask budget.

## Split invariant and atomic metering

For one budget and period let `L` be its allocation, `C` the sum of committed
child allocations, `S` its own settled spend, and `H` its live reservations.
Require `C + S + H <= L`; available quantity is `L - C - S - H`.
All quantities are checked nonnegative integers with checked arithmetic. Units
are requests, total input-plus-output tokens, or integer USD microdollars; no
floating-point balances or implicit currency conversion. Multiple required
budgets (for example requests and dollars) reserve together, all or nothing.

Splitting creates a child and moves available parent quantity into `C` in one
serializable durable transaction. Example: team 100, child A 40 and child B 30
leaves 30 for the team's own calls. A's spend consumes its 40, not another debit
of the parent's already committed 40. Recursion proves every descendant fits.
Parents and children share unit and period boundaries; a narrower validity
window is allowed. Different period cadences or currencies cannot be split.
A reduction cannot take back settled or held quantity. Revoking stops new holds;
reclaim only unspent, unheld quantity after outstanding calls settle. Past spend
remains committed in the parent until that period ends.

Reserve performs one transaction: authenticate the assignment revision, validate
its ancestry and validity, resolve the service clock's period, check every
required meter, insert a unique reservation, and increase `H`. A last-unit race
must yield one reservation and one Denied, across processes and hosts. Use a
shared transactional store with row locking or compare-and-swap plus unique
keys; an in-process mutex or cached read followed by a write is insufficient.
Split, revoke, reserve and settlement contend through the same transaction
boundary. Lock multiple meters in canonical ID order. Retry an internal CAS
collision against fresh state; report Conflict only if the operation cannot
complete, without calling the provider.

Settlement transitions one reservation once, removes its hold and adds its
charge to `S` in the same transaction as a durable log outbox record. Duplicate
settlement has no effect; an inconsistent duplicate is Conflict. Persist an
attempt's dispatch intent before transport I/O. After an ambiguous crash, never
resend automatically: reconcile or conservatively charge the hold. A reservation
can be released only with proof no billable dispatch occurred or a verified
zero charge. Expiring a worker lease is not proof the provider stopped billing.
Retain unresolved holds through restarts and period changes.

## Estimate, execute, settle

Validate typed inputs and authorization before billable I/O. Pin the selected
provider/model, tokenizer revision and trusted price schedule. Discovery may be
needed to obtain these; unpriced billable discovery is refused, while declared
free discovery may run without an inference debit. Selection's `cost` tier is
not a price. Reserve before every billable attempt, including fallback to a
new model, with its own price and bound. Keep one logical-request settlement
record that totals all attempts. Do not assume a 404 is free without the
provider's charging contract.

Use a conservative bound rather than an average estimate:

- Requests: one per dispatched completion attempt, including retries. The logical
  request's hold covers all allowed attempts, or obtains another hold before retry.
- Tokens: exact trusted input tokenization including system text and template
  overhead, plus an enforced output maximum. Require `max_tokens` or apply an
  explicit policy default, constrained by the remaining allowance. Include hidden
  reasoning tokens if billable. If a provider cannot bound them, refuse that unit.
- Dollars: round the input and maximum output charge up to integer microdollars
  using a pinned, trusted tariff that includes minimum charges and other fees.
  Unknown price or a provider unable to honor the bound cannot use a hard dollar budget.

The door may expose an estimate for planning, but only the conservative bound
admits a call. Refund the difference on trusted final usage. Missing, malformed
or untrusted usage settles at the hold, with `conservative` status. Do not count
UTF-8 bytes as tokens. A local backend can consume requests or tokens even if
its configured dollar charge is zero.

If an estimate was low but the reserved bound suffices, settle actual cost and
release the rest. If execution can approach the bound, obtain an atomic top-up
before authorizing further billable work, or stop while the existing bound
still covers it. This requires a provider that actually supports such control;
canceling a response reader alone does not prove generation stopped. Recommend
fixed enforced output bounds for the first release instead of speculative top-ups.

If a provider violates its bound and bills more than the hold, do not silently
charge another budget or exceed the caller's allocation. Settle the hold, freeze
that provider for hard-budget calls, and record the excess as an operator
liability outside the delegated pool. Return Conflict if no response has been
returned; otherwise terminate the stream with Conflict and persist the incident.
A hard provider-invoice ceiling cannot be promised without a trustworthy billing
bound; admission control alone cannot reverse a bill. Brian must accept this
boundary or exclude such providers.

Streaming needs a settlement owner that outlives the consumer. Parse trusted
usage separately from emitted text; handle EOF, `[DONE]`, cancellation, timeout,
body drop and process restart exactly once. Until authoritative usage is known,
keep the hold, or conservatively settle it. Never release on disconnect alone.
Provider failures preserve their existing classes after accounting; a denied
reservation sends no completion request.

## Periods, cache and logging

Use UTC fixed half-open intervals `[start, end)` from the budget service's clock,
with a configured anchor and duration. A reservation belongs to the admission
period even when settlement arrives later. Its assignment must be valid at
admission; revocation does not erase an in-flight liability. No rollover by
default. Reset means creating a new meter, never zeroing an old one. Provision
child allocations for the next period transactionally from approved schedules;
if the parent cannot fund them, refuse activation rather than mint quantity.
Old unresolved holds remain in the old period and cannot become new capacity.
Store time monotonically enough to prevent clock rollback reopening a period.

Cache immutable policy revisions only, keyed by authorization context; current
revocation and assignment revisions are checked live at reserve. Balances,
reservations, admission and settlement responses are uncacheable. Do not cache
away spending side effects. Ask remains uncacheable. For a future deterministic
inference cache, charge only provider execution on a miss and still check live
read authority on a hit; define any charge-per-served-request policy separately.
Embedding caching needs this same review before sharing the gate.

Write one structured ikigai-log line per settled logical request through the
transactional outbox. It contains budget IDs and period starts, the principal
already named by the capability, unit totals per budget across attempts, attempt
count, charged and released amounts, and a coarse outcome (`actual`,
`conservative`, `provider-bound-violation`). A zero-charge dispatched request
still settles; a pre-admission refusal creates no settlement line. Aggregate
counts only: no prompt, completion, token IDs, IP, device, user agent, provider
request ID, caller correlation ID or extra person identifier. No per-token or
per-attempt log lines. Approval audit data stays in proposals, not usage logs.

The internal random settlement key deduplicates outbox delivery; it is not an
exported tracking field. Exactly one visible log line requires an idempotent
sink that can acknowledge this key without adding it to the line. If ikigai-log
cannot do that, this is an integration prerequisite, not a claim about its
current API. Do not release an outbox record until acknowledged. Meter state,
not log reconstruction, is authoritative. Log outages retain the durable
outbox; if it cannot be persisted, refuse new reservations with Conflict.
Restrict reads and set a retention policy for both meters and logs; principal
and timestamps can still reveal activity even without content.

## Refusals and failures

These are proposed local policy errors; provider and transport errors keep their
existing classes. In particular, the existing origin failure declarations
include OverBudget for transport/body limits, not this allocation decision
([src/lib.rs:1247–1275](../src/lib.rs#L1247)). A spent capability is Denied.
All local refusals name the budget and the next useful action in plain text.

| Kind | Condition and example sentence |
|---|---|
| Denied | Missing spend authority, expired/revoked assignment, wrong principal/scope, or insufficient available quantity: “Budget `team-a` has 20 tokens available; this request needs a 30-token reservation. Reduce max_tokens or request an approved allocation.” An unauthorized caller gets only “Budget `team-a` is not available to this capability. Request spend access.” |
| Conflict | Stale proposal revision, concurrent edit that cannot commit, meter unavailable, incompatible repeated settlement, or provider-bound violation: “Budget `team-a` could not reserve quantity because its meter is unavailable. Retry when the meter is available.” Never fail open. |
| InvalidArgument | Malformed budget ID, ambiguous selection, incompatible units/periods, zero/negative allocation, arithmetic overflow, or no trusted cost bound: “Budget `team-a` cannot bound this model's token cost. Choose a model with an enforced token limit.” A well-formed split beyond available parent quantity is Denied, not a malformed input. |
| NotFound | An authorized reader names a missing budget, policy revision or reservation: “Budget `team-a` was not found. Choose an existing authorized budget.” Never reveal another principal's budget existence. |

Reserve and policy endpoints declare Denied and permanent failures; a transient
store outage is deliberately Conflict in this proposed policy API, without an
automatic replay of billable work. Ask must describe these additional local
failure possibilities alongside its existing origin failures. Streaming failures
must travel through the body error channel if headers have already been returned.

## Tests that would prove the implementation

Only the baseline test ships with this spike. The following are acceptance tests
for a future implementation, with fake clocks, tariffs and transports:

1. Within budget: reserve 30 of 100 tokens, return trusted usage 22, assert one
   dispatch, `S=22`, `H=0`, available 78, one settlement line, and no prompt in it.
2. Past budget: with available 20, request bound 30; assert Denied naming the
   budget, unchanged meter and zero transport calls for both direct and facade asks.
3. Split: allocate 60 of parent 100, then concurrently request two children of
   30 each; at most one succeeds. Property-test recursive `C+S+H<=L`, including
   reductions, revocation, overlapping validity and checked-integer boundaries.
4. Last-unit race: two doors on independent connections reserve the last request
   unit behind a barrier. Exactly one sends; the other is Denied. Repeat with
   many callers and a real shared store, not only a process-local mock.
5. Identity: copying or attenuating a capability shares its meter. Widened scope,
   forged principal, expired assignment and removed parent are refused. An ask
   caller cannot settle, approve its own widening, or write a meter directly.
6. Settlement: duplicate completion, restart after dispatch, log acknowledgment
   loss and concurrent reconciliation cannot double-charge, double-refund or
   duplicate a log line. Missing usage, disconnect and stream drop charge the hold.
7. Estimation: include system/template tokens, output/reasoning limits, rounded
   tariffs and fallback attempts. A low estimate within the hold refunds correctly;
   a violated bound freezes the provider and does not debit another budget.
8. Period: admission just before midnight settles in the old period after reset;
   no rollover or released old hold enlarges the next period. Test clock rollback,
   child schedule activation failure and validity ending during a stream.
9. Multi-unit admission: requests fit but dollars do not; no meter changes and no
   dispatch. Race reservation against split and revoke, with serializable outcomes.
10. Contracts and privacy: descriptions match enforced scopes and failure classes;
    unauthorized unknown/existing IDs are indistinguishable. Logs contain only
    allowed aggregates and the capability principal. Exercise embedding misses
    and any permitted cache hits so neither becomes an enforcement bypass.

## Migration and rollout

This spike changes no published endpoint, dependency or configuration format.
The future mandatory spend grant changes the published authorization contract:
existing network-only callers in the baseline will be refused. Release it with
an explicit compatibility notice and version decision, not silently as a patch.

Introduce durable meters and approved allocations first. Map existing assignments
to explicit budgets through proposals; do not infer amounts from flat scope
lists, create unlimited legacy grants, or rewrite all grants with a replacement
operation. Operator-selected shadow estimates may help sizing, but are not
protection and must not log content. Before enabling enforcement, validate
provider bounds, multi-door atomicity, crash recovery, the log sink's deduplication
contract and allocations for every active principal. Enable the gate on both
facade and direct endpoints together. If embedding is deferred, state the limited
scope and prevent roles from using it as an alternate unmetered inference path.
Rollback stops admission or retains metering with the previous compatible build;
it must not discard holds or restore unlimited access. Reconcile pending attempts
and preserve old period state across deployment.

## DECISIONS FOR BRIAN

1. **How quantity rides on authority.** Recommend an assignment-bound budget
   reference plus structured attenuation constraints, with counters outside core.
   Review core's actual capability/mount contracts before selecting the extension;
   change core in its own proposal only if these constraints cannot survive today.
2. **Where atomicity lives.** Recommend a host-injected shared transactional budget
   service, with one authoritative store across doors and all-or-nothing multi-unit
   reservations. Do not distribute independent local allowances initially.
3. **What a split means.** Recommend committed child allocations deducted from
   the parent's available pool, identical periods and units, no borrowing, and
   reclaim only demonstrably unspent quantity. This makes narrowing provable.
4. **Which quantities ship first.** Recommend requests and total tokens first;
   add USD microdollars only with trusted versioned tariffs and provider bounds.
   Do not treat routing cost tiers as prices.
5. **How hard the ceiling is.** Recommend enforced conservative bounds and no
   speculative streaming top-ups initially. Exclude providers without billing
   bounds from hard monetary budgets; assign bound violations to an explicit
   operator liability rather than overdrawing delegated authority.
6. **Periods and rollover.** Recommend fixed UTC anchored intervals, no rollover,
   admission-period settlement and independently retained old meters. Defer
   calendar-time-zone and rolling-window policies.
7. **Approvals and refusal classes.** Recommend proposals for all policy edits,
   distinct-person approval for widening, Denied for exhausted authority, and
   Conflict for meter availability/revision failures. Keep existing provider
   failure types and transport OverBudget behavior distinct.
8. **Logging guarantee and retention.** Recommend a transactional outbox and a
   deduplicating ikigai-log sink as a release prerequisite for one line per settled
   request. Approve an explicit retention period and principal-scoped read policy
   before rollout; never export internal settlement identifiers.
9. **Coverage and migration.** Recommend applying budgets to all billable inference
   entry points, including embedding misses, in the enforcement release. Budget
   grants become mandatory through a documented authorization migration; no
   indefinite network-only compatibility bypass.
