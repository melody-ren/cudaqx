/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// Wire types carried on decoding-task-graph edges. These shapes are part of
/// the graph contract: a streaming executor replaces the executor internals,
/// not these types.

/// @brief Raw measurement bits entering the graph, with a caller-chosen tag
/// that identifies the shot/window.
struct measurement_results {
  std::vector<std::uint8_t> bits;
  std::uint64_t tag = 0;
};

/// @brief Detection events (one entry per detector, 0.0/1.0 for hard
/// events). Produced by a `d_apply` node, carried on `delta` edges between
/// compiled tasking nodes, and accepted directly by run() for graphs whose
/// declared input is detection events. `converged` propagates decoder
/// convergence along detector-delta edges; it is true for raw inputs.
struct detection_events {
  std::vector<float_t> events;
  bool converged = true;
};

/// @brief Decoder output produced by a `decode` node, in the error basis.
struct error_pattern {
  bool converged = false;
  std::vector<float_t> result;
  std::optional<cudaqx::heterogeneous_map> opt_results;
};

/// @brief Logical observable value carried between `o_project`, `xor` and
/// `root` nodes, and returned per root from run().
struct logical_outcome {
  std::vector<std::uint8_t> bits;
  bool converged = false;
};

/// @brief A decoding task graph: topology as data, loaded from the
/// `dtg-kinds/v0` IR dialect.
///
/// Node kinds: `d_apply` (measurement bits -> detection events via D),
/// `decode` (detection events -> error pattern via a decoder plugin resolved
/// from `binding_ref` "decoder:<name>"), `o_project` (error pattern ->
/// logical bits via O, one output port `l<k>` per observable), `xor` (GF(2)
/// combine of two logical inputs) and `root` (sink, one per logical
/// observable). All model matrices (D, H, O) and the decoders themselves are
/// fixed at graph construction.
///
/// Graphs also load from a partitioned tasking bundle (see from_bundle),
/// which adds the compiled node kinds `ingest` (identity on the graph
/// input), `compiled_decode` (a solve: slices its declared detector domain
/// out of the global detection events, XORs in typed detector deltas, and
/// decodes with its own local DEM into a correction-candidate vector, one
/// candidate per local-DEM observable), `compiled_effect_view` (a sparse
/// GF(2) apply of one precomputed project-to-commit payload: candidate ->
/// logical contribution and, when the view owns boundary detectors, a
/// detector delta), `compiled_contribution` (a fused solve + effect view)
/// and `combine_xor` (variadic GF(2) fold of logical contributions). All of
/// these are likewise fixed at load: run() follows the wiring and decides
/// nothing.
///
/// Graph-input semantics: run()'s input is whatever the loaded graph
/// declares as its external input port. dtg-kinds/v0 graphs start at raw
/// measurements (a `d_apply` front), so they take measurement_results;
/// tasking bundles start at detection events (measurement-to-detector
/// conversion is deferred upstream), so they take detection_events. Calling
/// the wrong overload throws.
///
/// Canonical IR form: to_ir_json() re-emits exactly the keys and array
/// orderings that from_ir_json() loaded, so a load/re-emit round trip is
/// equal to the input modulo JSON object key order (numeric values are
/// preserved: integers as integers, floating-point via shortest
/// round-trippable form).
///
/// This is the synchronous walking-skeleton executor; the streaming
/// implementation replaces the internals behind this interface.
class decoding_task_graph {
public:
  /// @brief Load a graph from `dtg-kinds/v0` IR JSON. Constructs every
  /// `decode` node's decoder now. Malformed IR, unknown node kinds and
  /// unresolvable binding_refs throw std::runtime_error.
  static decoding_task_graph from_ir_json(const std::string &ir_json);

  /// @brief Load a graph from a `decoder-tasking-bundle/v1` directory.
  ///
  /// Reads `manifest.json`, verifies the SHA-256 (and size) of the program
  /// and of every inventoried artifact against the manifest, loads the
  /// `decoder-task-graph-ir/v1` program and resolves every `dem` and
  /// `project_view` artifact through its content-addressed URI (paths are
  /// confined to the bundle root). Any hash/size/inventory mismatch throws.
  /// Every `compiled_decode`/`compiled_contribution` node's decoder is
  /// constructed now, from that node's own local DEM (H, priors and
  /// observables), via decoder_init -> get_decoder(@p decoder_name)
  /// requesting observable output — the same binding the reference runtime
  /// applies. The default, "pymatching", matches the reference binder.
  static decoding_task_graph
  from_bundle(const std::string &bundle_dir,
              const std::string &decoder_name = "pymatching");

  /// @brief Execute one shot synchronously. Returns one logical_outcome per
  /// `root` node, ordered by the roots' `observable_index` param. Only valid
  /// for graphs whose declared input is raw measurements (a `d_apply`
  /// front); throws otherwise.
  std::vector<logical_outcome> run(const measurement_results &measurements);

  /// @brief Execute one shot on detection events. Only valid for graphs
  /// whose declared input is detection events (tasking bundles); throws
  /// otherwise. Returns one logical_outcome per external output, ordered as
  /// output_names().
  std::vector<logical_outcome> run(const detection_events &events);

  /// @brief Names of the graph's outputs, aligned with run()'s result. For
  /// bundle-loaded graphs these are the program's external output names in
  /// lexicographic order; for dtg-kinds/v0 graphs they are the root node ids
  /// ordered by observable_index.
  const std::vector<std::string> &output_names() const;

  /// @brief Re-emit the loaded IR in canonical form (see class docs). Only
  /// supported for graphs loaded from dtg-kinds/v0 IR; bundle-loaded graphs
  /// throw (the bundle on disk stays the canonical artifact).
  std::string to_ir_json() const;

  decoding_task_graph(const decoding_task_graph &) = default;
  decoding_task_graph(decoding_task_graph &&) noexcept = default;
  decoding_task_graph &operator=(const decoding_task_graph &) = default;
  decoding_task_graph &operator=(decoding_task_graph &&) noexcept = default;
  ~decoding_task_graph();

private:
  struct impl;
  explicit decoding_task_graph(std::shared_ptr<impl> state);
  std::shared_ptr<impl> impl_;
};

} // namespace cudaq::qec
