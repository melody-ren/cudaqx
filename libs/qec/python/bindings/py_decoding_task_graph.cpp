/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_decoding_task_graph.h"

#include "cudaq/qec/decoding_task_graph.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec {

void bindDecodingTaskGraph(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  nb::class_<logical_outcome>(qecmod, "LogicalOutcome", R"pbdoc(
    Logical observable value produced per graph root by
    DecodingTaskGraph.run(). `bits` holds the decoded logical bit(s);
    `converged` reports whether every decoder on the path converged.)pbdoc")
      .def_ro("bits", &logical_outcome::bits)
      .def_ro("converged", &logical_outcome::converged);

  nb::class_<decoding_task_graph>(qecmod, "DecodingTaskGraph", R"pbdoc(
    A decoding task graph loaded from dtg-kinds/v0 IR JSON. Topology,
    model matrices (D/H/O) and decoders are fixed at construction;
    run() executes one shot synchronously.)pbdoc")
      .def_static(
          "from_ir_json",
          [](const std::string &ir) {
            return decoding_task_graph::from_ir_json(ir);
          },
          nb::arg("ir"),
          "Load a graph from dtg-kinds/v0 IR JSON, constructing every decode "
          "node's decoder. Raises RuntimeError on malformed IR, unknown node "
          "kinds or unresolvable binding_refs.")
      .def(
          "run",
          [](decoding_task_graph &self, const std::vector<std::uint8_t> &bits,
             std::uint64_t tag) {
            return self.run(measurement_results{bits, tag});
          },
          nb::arg("bits"), nb::arg("tag") = 0,
          "Execute one shot on raw measurement bits. Returns one "
          "LogicalOutcome per root node, ordered by the roots' "
          "observable_index.")
      .def(
          "to_ir_json",
          [](const decoding_task_graph &self) { return self.to_ir_json(); },
          "Re-emit the loaded IR in canonical form (equal to the input "
          "modulo JSON object key order).");
}

} // namespace cudaq::qec
