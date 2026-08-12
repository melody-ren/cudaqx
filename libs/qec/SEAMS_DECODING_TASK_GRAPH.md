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

- C++: `cudaq::qec::decoding_task_graph::{from_ir_json, run, to_ir_json}` with
  wire types `measurement_results{bits, tag}` → … → `logical_outcome{bits,
  converged}` (see header).
- Python: `cudaq_qec.DecodingTaskGraph.from_ir_json(str)`,
  `.run(bits, tag=0) -> list[LogicalOutcome]`, `.to_ir_json()`.

## Verification shipped with the stub

- `unittests/test_decoding_task_graph.cpp` (5 groups): IR round-trip,
  **parity vs direct decoder + manual D/O** (the graph is a faithful
  re-plumbing), two-root + xor fan-in, 14 error paths, fixture load/run.
- `unittests/dtg_data/*.json`: two self-contained fixtures emitted by the IR
  reference serde — a monolithic d=5 memory and a **two-root fixture from a
  real disjoint-component decomposition**. An external harness holds the stub
  to 1000/1000 shot-exact agreement against two independent references on
  both fixtures.

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
