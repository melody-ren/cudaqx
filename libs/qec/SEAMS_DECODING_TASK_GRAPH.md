# decoding_task_graph stub — contract seams (handoff)

This branch carries a deliberately thin, working `decoding_task_graph`. The
implementation is scaffolding; the **seams below are the deliverable** and are
what the end-to-end verification pins. The real (streaming) implementation is
expected to replace the internals freely while keeping these seams.

## The five seams

1. **Topology is data.** Nodes/edges/ports are held as a serializable
   structure (`decoding_task_graph.h`, behind a pimpl). The stub executor is a
   synchronous topological walk supporting chains, fan-out, and xor fan-in.
2. **Graphs load from IR.** `from_ir_json()` accepts the `dtg-kinds/v0`
   dialect (kinds: `d_apply`, `decode`, `o_project`, `xor`, `root`) of the
   decoder task-graph IR (schema `decoder-task-graph-ir/v1`); `to_ir_json()`
   round-trips with a fixed point on re-emit. Kernels are never serialized —
   `decode` nodes carry a `binding_ref` and rebind at load.
3. **Roots are plural.** `run()` returns one `logical_outcome` per `root`
   node, ordered by `observable_index`. The graph unit is one fault-connected
   component; a component determines as many observables as it has roots.
4. **Decoder-agnostic construction.** `binding_ref: "decoder:<name>"`
   resolves through the plugin registry via `decoder_init` → `get_decoder`.
   No decoder type is special-cased; no `set_D_sparse`/`set_O_sparse`
   anywhere in this surface.
5. **D/O fixed at construction.** `d_apply` carries `d_sparse`, `o_project`
   carries `o_sparse`, decoders get H/priors — all as construction-time node
   parameters. After `from_ir_json`, `run()` decides nothing.

## Interface (the e2e harness is written against exactly this)

- C++: `cudaq::qec::decoding_task_graph::{from_ir_json, from_bundle, run,
  output_names, to_ir_json}` with wire types `measurement_results{bits,
  tag}` / `detection_events{events, converged}` → … →
  `logical_outcome{bits, converged}` (see header).
- Python: `cudaq_qec.DecodingTaskGraph.from_ir_json(str)`,
  `.from_bundle(dir, decoder="pymatching")`,
  `.run(bits, tag=0) -> list[LogicalOutcome]`,
  `.run_detection_events(events)`, `.output_names()`, `.to_ir_json()`.

## Tasking bundles (partitioned compiled programs)

`from_bundle(dir)` loads a `decoder-tasking-bundle/v1` directory: it reads
`manifest.json`, verifies the SHA-256 and size of the program and of every
inventoried artifact (fail-loud on any mismatch, including audit sidecars),
loads the `decoder-task-graph-ir/v1` program, and resolves each `dem` /
`project_view` artifact through its content-addressed URI, confined to the
bundle root. Pass-by-path is the contract: payloads are never inlined into
a mega-JSON.

Supported task `binding_ref`s: `ingest` (identity), `compiled_decode` (a
solve — slices `partition.domain_detectors` out of the global detection
events, XORs typed detector deltas in at their port-binding positions, and
decodes with its own local DEM into one candidate bit per local-DEM
observable), `compiled_effect_view` (sparse GF(2) apply of a
`decoder-tasking-project-to-commit/v1` payload: candidate → `L` logical
contribution and, when the view owns boundary detectors, a `delta`),
`compiled_contribution` (fused solve + view) and `combine_xor` /`xor`
(GF(2) fold to the external outputs). Solve decoders are constructed at
load via decoder_init → get_decoder with observable output; the default
`pymatching` binding matches the reference runtime (local DEMs are parsed
with decomposition suggestions expanded, mirroring PyMatching's
from_detector_error_model). Everything is fixed at load; run() follows the
wiring and decides nothing. Unsupported program features (composite task
bodies, guards, solve-owned logical outputs, logical-from-solve views)
fail loudly at load, matching the Stage-1 reference binder.

These bundles start at detection events — measurement-to-detector
conversion is deferred upstream — so run()'s input is the graph's declared
input port: `detection_events` here, `measurement_results` when a
`d_apply` front exists. Roots are synthesized from `external_outputs`, one
per output name in lexicographic order (`output_names()` gives the
alignment). Bundle-loaded graphs do not re-emit IR: the bundle directory
on disk stays the canonical artifact.

## Verification shipped with the stub

- `unittests/test_decoding_task_graph.cpp` (7 groups): IR round-trip,
  **parity vs direct decoder + manual D/O** (the graph is a faithful
  re-plumbing), two-root + xor fan-in, 14 error paths, fixture load/run,
  bundle load + hand-computed per-kind semantics, and bundle tamper
  detection.
- `unittests/dtg_data/*.json`: two self-contained fixtures emitted by the IR
  reference serde — a monolithic d=5 memory and a **two-root fixture from a
  real disjoint-component decomposition**. An external harness holds the stub
  to 1000/1000 shot-exact agreement against two independent references on
  both fixtures.
- `unittests/dtg_data/mini_bundle/`: a hand-written synthetic
  `decoder-tasking-bundle/v1` (two solves, three views, one detector-delta
  handoff, a fused contribution, a variadic xor, two external outputs)
  exercising every compiled node kind with hand-computed expectations, plus
  manifest tamper tests. An external harness additionally holds
  `from_bundle` to shot-exact agreement against the reference bundle
  runtime on a real 63-task partitioned program.

## Base statement

Branch base = main + "lift O/D into construction" + "dynamic DEM" (merged and
verified here), plus a local composition of the two in
`resolve_decoder_init`. That composition is **superseded** by the upstream
`decoder_init::from_dem_chunks` work — intentionally not chased here; the
stub class does not depend on it. When rebasing onto the current dynamic-DEM
head, drop this branch's `realtime_decoding.cpp` resolution in favor of
upstream.

## Known placeholders (replace freely)

Synchronous per-shot interpreter; `tag` plumbed but unused (no
windowing/reordering); host-side XOR loops for `d_apply`/`o_project`/`xor`;
one `decode()` call per shot per node; detector-count inference assumes a
`d_apply` producer; `error_pattern.opt_results` propagated but unconsumed.
