/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoding_task_graph.h"

#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;
using cudaq::qec::decoding_task_graph;
using cudaq::qec::measurement_results;

namespace {

/// Tiny hand-written dtg-kinds/v0 program over a [3,1] repetition-style
/// model: 4 raw measurements -> 2 detectors -> 3 error mechanisms -> 1
/// logical observable, decoded by the built-in single_error_lut decoder.
///
/// D = [[0,1],[1,2]] (m3 is a deliberately unused measurement)
/// H = det x mech = {mech0 -> det0, mech1 -> det0+det1, mech2 -> det1}
/// O = obs0 flips on mech0 or mech2
const char *tiny_ir = R"({
  "schema_version": "dtg-kinds/v0",
  "derived_from": "decoder-task-graph-ir/v1",
  "metadata": {"program": "tiny_rep", "seed": 7},
  "nodes": [
    {"id": "dets", "kind": "d_apply", "inputs": ["m"], "outputs": ["s"],
     "params": {"num_measurements": 4, "num_detectors": 2,
                "d_sparse": [[0, 1], [1, 2]]}},
    {"id": "dec", "kind": "decode", "inputs": ["s"], "outputs": ["e"],
     "binding_ref": "decoder:single_error_lut",
     "params": {"detector_domain": [0, 1], "num_mechanisms": 3,
                "h_sparse": [[0], [0, 1], [1]],
                "priors": [0.01, 0.01, 0.01]}},
    {"id": "obs", "kind": "o_project", "inputs": ["e"], "outputs": ["l0"],
     "params": {"num_observables": 1, "o_sparse": [[0, 2]]}},
    {"id": "root_0", "kind": "root", "inputs": ["l"], "outputs": [],
     "params": {"observable_index": 0}}
  ],
  "edges": [
    ["dets", "s", "dec", "s"],
    ["dec", "e", "obs", "e"],
    ["obs", "l0", "root_0", "l"]
  ],
  "inputs": {"m": ["dets", "m"]},
  "roots": ["root_0"]
})";

/// Two roots plus an xor combine: obs projects two observables; root_0 takes
/// l0, root_1 takes l0 XOR l1. Roots are listed out of observable order to
/// prove run() orders by observable_index, not list order.
const char *two_root_ir = R"({
  "schema_version": "dtg-kinds/v0",
  "nodes": [
    {"id": "dets", "kind": "d_apply", "inputs": ["m"], "outputs": ["s"],
     "params": {"num_measurements": 4, "num_detectors": 2,
                "d_sparse": [[0, 1], [1, 2]]}},
    {"id": "dec", "kind": "decode", "inputs": ["s"], "outputs": ["e"],
     "binding_ref": "decoder:single_error_lut",
     "params": {"detector_domain": [0, 1], "num_mechanisms": 3,
                "h_sparse": [[0], [0, 1], [1]],
                "priors": [0.01, 0.01, 0.01]}},
    {"id": "obs", "kind": "o_project", "inputs": ["e"],
     "outputs": ["l0", "l1"],
     "params": {"num_observables": 2, "o_sparse": [[0, 2], [2]]}},
    {"id": "combine", "kind": "xor", "inputs": ["a", "b"], "outputs": ["l"]},
    {"id": "root_1", "kind": "root", "inputs": ["l"], "outputs": [],
     "params": {"observable_index": 1}},
    {"id": "root_0", "kind": "root", "inputs": ["l"], "outputs": [],
     "params": {"observable_index": 0}}
  ],
  "edges": [
    ["dets", "s", "dec", "s"],
    ["dec", "e", "obs", "e"],
    ["obs", "l0", "root_0", "l"],
    ["obs", "l0", "combine", "a"],
    ["obs", "l1", "combine", "b"],
    ["combine", "l", "root_1", "l"]
  ],
  "inputs": {"m": ["dets", "m"]},
  "roots": ["root_1", "root_0"]
})";

std::string read_file(const std::string &path) {
  std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "cannot open " << path;
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

json patched(const char *base, void (*mutate)(json &)) {
  auto doc = json::parse(base);
  mutate(doc);
  return doc;
}

} // namespace

// (a) IR round-trip: what from_ir_json loads, to_ir_json re-emits equal
// modulo JSON object key order (== nlohmann value equality).
TEST(DecodingTaskGraphTest, IrRoundTrip) {
  auto g = decoding_task_graph::from_ir_json(tiny_ir);
  auto emitted = g.to_ir_json();
  EXPECT_EQ(json::parse(emitted), json::parse(tiny_ir));

  // The re-emitted form is itself loadable and canonical (a fixed point).
  auto g2 = decoding_task_graph::from_ir_json(emitted);
  EXPECT_EQ(json::parse(g2.to_ir_json()), json::parse(emitted));
}

// (b) Parity against a directly constructed decoder with manual D-apply and
// O-projection on the same model.
TEST(DecodingTaskGraphTest, ParityWithDirectDecoder) {
  auto g = decoding_task_graph::from_ir_json(tiny_ir);

  auto H = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      2, 3, {{0}, {0, 1}, {1}});
  auto dec = cudaq::qec::get_decoder(
      "single_error_lut",
      cudaq::qec::decoder_init(H, std::nullopt, {0.01, 0.01, 0.01}));

  const std::vector<std::vector<std::uint8_t>> shots = {
      {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0},
      {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 1}, {0, 1, 1, 0}};
  for (const auto &bits : shots) {
    // Manual reference: D-apply, decode, O-project.
    std::vector<cudaq::qec::float_t> syndrome = {
        static_cast<cudaq::qec::float_t>(bits[0] ^ bits[1]),
        static_cast<cudaq::qec::float_t>(bits[1] ^ bits[2])};
    auto ref = dec->decode(syndrome);
    std::uint8_t expected_bit = (ref.result[0] >= 0.5 ? 1 : 0) ^
                                (ref.result[2] >= 0.5 ? 1 : 0);

    auto outcomes = g.run(measurement_results{bits, 0});
    ASSERT_EQ(outcomes.size(), 1u);
    ASSERT_EQ(outcomes[0].bits.size(), 1u);
    EXPECT_EQ(outcomes[0].bits[0], expected_bit) << "shot mismatch";
    EXPECT_EQ(outcomes[0].converged, ref.converged);
  }
}

// (c) Two roots ordered by observable_index (roots listed out of order in
// the IR), with an xor fan-in on the second root's path.
TEST(DecodingTaskGraphTest, TwoRootsOrderedByObservableIndexWithXor) {
  auto g = decoding_task_graph::from_ir_json(two_root_ir);

  // m = {0,1,1,0}: detectors = (0^1, 1^1) = (1,0) -> single_error_lut picks
  // mechanism 0 -> l0 = 1 (mech 0 in O row 0), l1 = 0 (row 1 is {mech 2}).
  auto outcomes = g.run(measurement_results{{0, 1, 1, 0}, 42});
  ASSERT_EQ(outcomes.size(), 2u);
  EXPECT_EQ(outcomes[0].bits, std::vector<std::uint8_t>{1}); // observable 0
  EXPECT_EQ(outcomes[1].bits, std::vector<std::uint8_t>{1}); // l0 ^ l1 = 1^0
  EXPECT_TRUE(outcomes[0].converged);
  EXPECT_TRUE(outcomes[1].converged);

  // m = {0,0,1,0}: detectors = (0,1) -> mechanism 2 -> l0 = 1, l1 = 1,
  // so root_1 (xor) = 0.
  outcomes = g.run(measurement_results{{0, 0, 1, 0}, 43});
  ASSERT_EQ(outcomes.size(), 2u);
  EXPECT_EQ(outcomes[0].bits, std::vector<std::uint8_t>{1});
  EXPECT_EQ(outcomes[1].bits, std::vector<std::uint8_t>{0});

  // Round-trip holds for the two-root program as well.
  EXPECT_EQ(json::parse(g.to_ir_json()), json::parse(two_root_ir));
}

// (d) Error paths: malformed IR, unknown kinds, bad bindings, bad wiring and
// bad runtime input must all fail loudly.
TEST(DecodingTaskGraphTest, ErrorPaths) {
  // Malformed JSON.
  EXPECT_THROW(decoding_task_graph::from_ir_json("{"), std::runtime_error);
  // Wrong top-level type.
  EXPECT_THROW(decoding_task_graph::from_ir_json("[1,2]"), std::runtime_error);
  // Wrong schema version.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) { d["schema_version"] = "dtg/v9"; })
                       .dump()),
               std::runtime_error);
  // Unknown top-level key.
  EXPECT_THROW(
      decoding_task_graph::from_ir_json(
          patched(tiny_ir, [](json &d) { d["mystery"] = 1; }).dump()),
      std::runtime_error);
  // Unknown node kind.
  EXPECT_THROW(
      decoding_task_graph::from_ir_json(
          patched(tiny_ir, [](json &d) { d["nodes"][0]["kind"] = "d_undo"; })
              .dump()),
      std::runtime_error);
  // binding_ref without the decoder: prefix.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) {
                             d["nodes"][1]["binding_ref"] = "single_error_lut";
                           })
                       .dump()),
               std::runtime_error);
  // Unknown decoder plugin.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) {
                             d["nodes"][1]["binding_ref"] =
                                 "decoder:no_such_decoder_plugin";
                           })
                       .dump()),
               std::runtime_error);
  // binding_ref on a non-decode node.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) {
                             d["nodes"][0]["binding_ref"] = "decoder:x";
                           })
                       .dump()),
               std::runtime_error);
  // h_sparse index out of the detector_domain range.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) {
                             d["nodes"][1]["params"]["h_sparse"][0] = {7};
                           })
                       .dump()),
               std::runtime_error);
  // priors length mismatch.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) {
                             d["nodes"][1]["params"]["priors"] = {0.01};
                           })
                       .dump()),
               std::runtime_error);
  // Unfed input port (dropped edge).
  EXPECT_THROW(
      decoding_task_graph::from_ir_json(
          patched(tiny_ir, [](json &d) { d["edges"].erase(1); }).dump()),
      std::runtime_error);
  // Input port fed twice.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(tiny_ir,
                           [](json &d) {
                             d["edges"].push_back({"dec", "e", "obs", "e"});
                           })
                       .dump()),
               std::runtime_error);
  // Duplicate root observable_index.
  EXPECT_THROW(
      decoding_task_graph::from_ir_json(
          patched(two_root_ir,
                  [](json &d) {
                    d["nodes"][4]["params"]["observable_index"] = 0;
                  })
              .dump()),
      std::runtime_error);
  // Cycle between two xor nodes.
  EXPECT_THROW(decoding_task_graph::from_ir_json(
                   patched(two_root_ir,
                           [](json &d) {
                             d["nodes"].push_back(
                                 {{"id", "loop"},
                                  {"kind", "xor"},
                                  {"inputs", {"a", "b"}},
                                  {"outputs", {"l"}}});
                             // combine.a now comes from loop, and loop feeds
                             // on combine: a cycle.
                             d["edges"][3] = {"loop", "l", "combine", "a"};
                             d["edges"].push_back(
                                 {"combine", "l", "loop", "a"});
                             d["edges"].push_back({"obs", "l0", "loop", "b"});
                           })
                       .dump()),
               std::runtime_error);

  // Runtime: wrong measurement width.
  auto g = decoding_task_graph::from_ir_json(tiny_ir);
  EXPECT_THROW(g.run(measurement_results{{1, 0}, 0}), std::runtime_error);
}

// (e) Fixture smoke: load the checked-in dtg-kinds/v0 fixtures (pymatching
// binding) and run one trivial all-zero shot through each.
TEST(DecodingTaskGraphTest, FixtureSmokeSingleShot) {
  struct fixture_case {
    const char *file;
    std::size_t num_measurements;
    std::size_t num_roots;
  };
  for (const auto &fc :
       {fixture_case{"/monolithic_d5.json", 265, 1},
        fixture_case{"/two_root_d5.json", 530, 2}}) {
    auto ir = read_file(std::string(TEST_DATA_DIR) + fc.file);
    auto g = decoding_task_graph::from_ir_json(ir);

    // Canonical round-trip on real producer output.
    EXPECT_EQ(json::parse(g.to_ir_json()), json::parse(ir)) << fc.file;

    auto outcomes =
        g.run(measurement_results{std::vector<std::uint8_t>(
                                      fc.num_measurements, 0),
                                  0});
    ASSERT_EQ(outcomes.size(), fc.num_roots) << fc.file;
    for (const auto &oc : outcomes) {
      ASSERT_EQ(oc.bits.size(), 1u);
      EXPECT_EQ(oc.bits[0], 0) << fc.file;
      EXPECT_TRUE(oc.converged) << fc.file;
    }
  }
}
