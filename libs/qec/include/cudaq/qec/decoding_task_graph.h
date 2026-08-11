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

/// @brief Detection events produced by a `d_apply` node (one entry per
/// detector, 0.0/1.0 for hard events).
struct detection_events {
  std::vector<float_t> events;
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

  /// @brief Execute one shot synchronously. Returns one logical_outcome per
  /// `root` node, ordered by the roots' `observable_index` param.
  std::vector<logical_outcome> run(const measurement_results &measurements);

  /// @brief Re-emit the loaded IR in canonical form (see class docs).
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
