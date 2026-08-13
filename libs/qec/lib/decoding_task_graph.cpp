/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <tuple>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>

using json = nlohmann::json;

namespace cudaq::qec {

namespace {

constexpr std::string_view schema_version_v0 = "dtg-kinds/v0";
constexpr std::string_view decoder_binding_prefix = "decoder:";

// Partitioned tasking-bundle dialect.
constexpr std::string_view bundle_schema_version = "decoder-tasking-bundle/v1";
constexpr std::string_view program_schema_version = "decoder-task-graph-ir/v1";
constexpr std::string_view project_view_schema_version =
    "decoder-tasking-project-to-commit/v1";

/// The compiled_* kinds come from the tasking-bundle program dialect; the
/// first five are the dtg-kinds/v0 dialect. gf2_xor serves both `xor`
/// (binary, v0) and `combine_xor` (variadic, bundle).
enum class node_kind {
  d_apply,
  decode,
  o_project,
  gf2_xor,
  root,
  ingest,
  compiled_decode,
  compiled_effect_view,
  compiled_contribution
};

node_kind parse_kind(const std::string &kind, const std::string &node_id) {
  if (kind == "d_apply")
    return node_kind::d_apply;
  if (kind == "decode")
    return node_kind::decode;
  if (kind == "o_project")
    return node_kind::o_project;
  if (kind == "xor")
    return node_kind::gf2_xor;
  if (kind == "root")
    return node_kind::root;
  throw std::runtime_error("dtg: node '" + node_id + "' has unknown kind '" +
                           kind + "'");
}

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error("dtg: " + msg);
}

// ---- SHA-256 (FIPS 180-4), for bundle integrity verification -------------
// Self-contained so the qec library gains no crypto dependency for the sake
// of one content-addressing check.
class sha256 {
public:
  void update(const std::uint8_t *data, std::size_t len) {
    total_ += len;
    while (len > 0) {
      const std::size_t take = std::min(len, std::size_t{64} - fill_);
      std::memcpy(block_.data() + fill_, data, take);
      fill_ += take;
      data += take;
      len -= take;
      if (fill_ == 64) {
        compress();
        fill_ = 0;
      }
    }
  }

  std::string hex_digest() {
    const std::uint64_t bit_len = total_ * 8;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero = 0;
    while (fill_ != 56)
      update(&zero, 1);
    // The length bytes must not be counted in total_, but update() already
    // finished all data; feed them straight into the block.
    for (int i = 7; i >= 0; --i)
      block_[fill_++] = static_cast<std::uint8_t>(bit_len >> (8 * i));
    compress();
    std::string out;
    out.reserve(64);
    static const char *digits = "0123456789abcdef";
    for (auto word : h_)
      for (int i = 3; i >= 0; --i) {
        auto byte = static_cast<std::uint8_t>(word >> (8 * i));
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0xf]);
      }
    return out;
  }

private:
  static std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void compress() {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
        0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
        0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
        0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::array<std::uint32_t, 64> w;
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(block_[4 * i]) << 24) |
             (std::uint32_t(block_[4 * i + 1]) << 16) |
             (std::uint32_t(block_[4 * i + 2]) << 8) |
             std::uint32_t(block_[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] =
        std::tuple(h_[0], h_[1], h_[2], h_[3], h_[4], h_[5], h_[6], h_[7]);
    for (int i = 0; i < 64; ++i) {
      const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const auto ch = (e & f) ^ (~e & g);
      const auto t1 = h + s1 + ch + k[i] + w[i];
      const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const auto maj = (a & b) ^ (a & c) ^ (b & c);
      const auto t2 = s0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }

  std::array<std::uint32_t, 8> h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                     0xa54ff53a, 0x510e527f, 0x9b05688c,
                                     0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> block_{};
  std::size_t fill_ = 0;
  std::uint64_t total_ = 0;
};

std::string sha256_hex(const std::string &payload) {
  sha256 h;
  h.update(reinterpret_cast<const std::uint8_t *>(payload.data()),
           payload.size());
  return h.hex_digest();
}

/// One edge as loaded: [src_node, src_port, dst_node, dst_port].
struct edge {
  std::string src_node, src_port, dst_node, dst_port;
};

struct node {
  std::string id;
  node_kind kind = node_kind::root;
  std::vector<std::string> input_ports;
  std::vector<std::string> output_ports;
  /// Loaded verbatim and re-emitted verbatim; the typed fields below are
  /// projections of it.
  json params;
  bool has_params = false;
  std::string binding_ref; // decode only

  // d_apply
  std::size_t num_measurements = 0;
  std::vector<std::vector<std::uint32_t>> d_rows; // per detector

  // decode
  std::size_t num_mechanisms = 0;
  std::vector<std::uint32_t> detector_domain; // empty = full input
  std::shared_ptr<decoder> dec;

  // o_project
  std::size_t num_observables = 0;
  std::vector<std::vector<std::uint32_t>> o_rows; // per observable

  // root
  std::size_t observable_index = 0;

  // compiled_decode / compiled_contribution: global detector indices this
  // solve sees (order defines the local detector index), and, per input
  // slot, the local positions its typed detector delta scatters into (empty
  // for the syndrome slot itself).
  std::vector<std::uint32_t> compiled_domain;
  std::vector<std::vector<std::uint32_t>> delta_positions;
  std::size_t syndrome_slot = 0;
  std::size_t candidate_count = 0;

  // compiled_effect_view / compiled_contribution: sparse GF(2) rows of the
  // project-to-commit payload, indexing candidates.
  std::vector<std::vector<std::uint32_t>> logical_rows;
  std::vector<std::vector<std::uint32_t>> detector_rows;
  bool has_delta_output = false;

  /// Producer of each input port, aligned with input_ports. producer < 0
  /// means the port is fed by a graph input.
  std::vector<std::pair<int, std::string>> feeds;
};

const json &require_key(const json &obj, const char *key,
                        const std::string &ctx) {
  auto it = obj.find(key);
  if (it == obj.end())
    fail(ctx + " is missing required key '" + key + "'");
  return *it;
}

std::vector<std::string> string_list(const json &arr, const std::string &ctx) {
  if (!arr.is_array())
    fail(ctx + " must be an array of strings");
  std::vector<std::string> out;
  for (const auto &e : arr) {
    if (!e.is_string())
      fail(ctx + " must be an array of strings");
    out.push_back(e.get<std::string>());
  }
  return out;
}

std::size_t size_param(const json &params, const char *key,
                       const std::string &ctx) {
  const auto &v = require_key(params, key, ctx + " params");
  if (!v.is_number_unsigned())
    fail(ctx + " param '" + std::string(key) +
         "' must be a non-negative integer");
  return v.get<std::size_t>();
}

/// Parse a nested sparse index list: outer array of rows, each an array of
/// non-negative integers < bound.
std::vector<std::vector<std::uint32_t>>
nested_index_list(const json &arr, std::size_t expected_rows,
                  std::size_t index_bound, const std::string &ctx) {
  if (!arr.is_array())
    fail(ctx + " must be a nested array of indices");
  if (arr.size() != expected_rows)
    fail(ctx + " has " + std::to_string(arr.size()) + " rows, expected " +
         std::to_string(expected_rows));
  std::vector<std::vector<std::uint32_t>> out(arr.size());
  for (std::size_t r = 0; r < arr.size(); ++r) {
    const auto &row = arr[r];
    if (!row.is_array())
      fail(ctx + " row " + std::to_string(r) + " must be an array of indices");
    out[r].reserve(row.size());
    for (const auto &e : row) {
      if (!e.is_number_unsigned())
        fail(ctx + " row " + std::to_string(r) +
             " contains a non-integer index");
      auto idx = e.get<std::uint64_t>();
      if (idx >= index_bound)
        fail(ctx + " row " + std::to_string(r) + " index " +
             std::to_string(idx) + " is out of range (bound " +
             std::to_string(index_bound) + ")");
      out[r].push_back(static_cast<std::uint32_t>(idx));
    }
  }
  return out;
}

void reject_unknown_keys(const json &obj,
                         const std::vector<std::string> &allowed,
                         const std::string &ctx) {
  for (const auto &item : obj.items())
    if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
      fail(ctx + " has unknown key '" + item.key() + "'");
}

/// Correction candidate produced by a `compiled_decode` node: one bit per
/// local-DEM observable (the solve's pre-ownership candidate basis).
/// Internal wire value only; effect views turn it into logical outcomes and
/// detector deltas.
struct correction_candidate {
  std::vector<std::uint8_t> bits;
  bool converged = false;
};

/// Value flowing on a port during synchronous execution.
using port_value = std::variant<measurement_results, detection_events,
                                error_pattern, logical_outcome,
                                correction_candidate>;

/// Static type of a value on a port, used to type-check edges at load time.
/// The v0 dialect derives it from the node kind below; the bundle dialect
/// carries declared port types in the IR and is checked from those instead.
enum class port_type { measurements, detections, errors, logical };

port_type input_type(node_kind k) {
  switch (k) {
  case node_kind::d_apply:
    return port_type::measurements;
  case node_kind::decode:
    return port_type::detections;
  case node_kind::o_project:
    return port_type::errors;
  case node_kind::gf2_xor:
  case node_kind::root:
    return port_type::logical;
  default:
    break;
  }
  fail("node kind has no v0 port typing");
}

port_type output_type(node_kind k) {
  switch (k) {
  case node_kind::d_apply:
    return port_type::detections;
  case node_kind::decode:
    return port_type::errors;
  case node_kind::o_project:
  case node_kind::gf2_xor:
    return port_type::logical;
  default:
    break;
  }
  fail("node kind has no v0 output typing");
}

} // namespace

struct decoding_task_graph::impl {
  // Retained IR envelope for canonical re-emission.
  bool has_derived_from = false;
  std::string derived_from;
  bool has_metadata = false;
  json metadata;

  std::vector<node> nodes;
  std::unordered_map<std::string, std::size_t> node_index;
  std::vector<edge> edges;
  // Graph inputs, in load order: name -> (node, port).
  std::vector<std::tuple<std::string, std::string, std::string>> graph_inputs;
  std::vector<std::string> root_ids; // load order, for re-emission
  std::vector<std::size_t> topo_order;
  // Root node indices ordered by observable_index.
  std::vector<std::size_t> roots_by_observable;

  // What the graph's declared external input carries; selects which run()
  // overload is valid.
  enum class input_kind { measurements, detections };
  input_kind graph_input_kind = input_kind::measurements;
  // Bundle-loaded graphs have no v0 IR to re-emit.
  bool loaded_from_bundle = false;
  // Output names aligned with roots_by_observable.
  std::vector<std::string> output_names;
};

decoding_task_graph::decoding_task_graph(std::shared_ptr<impl> state)
    : impl_(std::move(state)) {}
decoding_task_graph::~decoding_task_graph() = default;

decoding_task_graph
decoding_task_graph::from_ir_json(const std::string &ir_json) {
  json doc;
  try {
    doc = json::parse(ir_json);
  } catch (const json::parse_error &e) {
    fail(std::string("IR is not valid JSON: ") + e.what());
  }
  if (!doc.is_object())
    fail("IR top level must be a JSON object");
  reject_unknown_keys(doc,
                      {"schema_version", "derived_from", "metadata", "nodes",
                       "edges", "inputs", "roots"},
                      "IR top level");

  const auto &schema = require_key(doc, "schema_version", "IR top level");
  if (!schema.is_string() || schema.get<std::string>() != schema_version_v0)
    fail("unsupported schema_version (expected '" +
         std::string(schema_version_v0) + "')");

  auto state = std::make_shared<impl>();
  if (auto it = doc.find("derived_from"); it != doc.end()) {
    if (!it->is_string())
      fail("'derived_from' must be a string");
    state->has_derived_from = true;
    state->derived_from = it->get<std::string>();
  }
  if (auto it = doc.find("metadata"); it != doc.end()) {
    state->has_metadata = true;
    state->metadata = *it;
  }

  // ---- Nodes -------------------------------------------------------------
  const auto &nodes_j = require_key(doc, "nodes", "IR top level");
  if (!nodes_j.is_array() || nodes_j.empty())
    fail("'nodes' must be a non-empty array");
  for (const auto &nj : nodes_j) {
    if (!nj.is_object())
      fail("every node must be a JSON object");
    node n;
    const auto &id_j = require_key(nj, "id", "node");
    if (!id_j.is_string())
      fail("node 'id' must be a string");
    n.id = id_j.get<std::string>();
    const std::string ctx = "node '" + n.id + "'";
    reject_unknown_keys(
        nj, {"id", "kind", "inputs", "outputs", "params", "binding_ref"}, ctx);
    const auto &kind_j = require_key(nj, "kind", ctx);
    if (!kind_j.is_string())
      fail(ctx + " 'kind' must be a string");
    n.kind = parse_kind(kind_j.get<std::string>(), n.id);
    n.input_ports = string_list(require_key(nj, "inputs", ctx), ctx + " inputs");
    n.output_ports =
        string_list(require_key(nj, "outputs", ctx), ctx + " outputs");
    if (auto it = nj.find("params"); it != nj.end()) {
      if (!it->is_object())
        fail(ctx + " 'params' must be an object");
      n.has_params = true;
      n.params = *it;
    } else {
      n.params = json::object();
    }
    if (auto it = nj.find("binding_ref"); it != nj.end()) {
      if (n.kind != node_kind::decode)
        fail(ctx + ": only 'decode' nodes may carry a binding_ref");
      if (!it->is_string())
        fail(ctx + " 'binding_ref' must be a string");
      n.binding_ref = it->get<std::string>();
    }

    // Arity by kind.
    const std::size_t want_in = n.kind == node_kind::gf2_xor ? 2 : 1;
    if (n.input_ports.size() != want_in)
      fail(ctx + " must declare exactly " + std::to_string(want_in) +
           " input port(s)");
    if (n.kind == node_kind::root) {
      if (!n.output_ports.empty())
        fail(ctx + ": root nodes must declare no output ports");
    } else if (n.kind != node_kind::o_project && n.output_ports.size() != 1)
      fail(ctx + " must declare exactly 1 output port");

    if (!state->node_index.emplace(n.id, state->nodes.size()).second)
      fail("duplicate node id '" + n.id + "'");
    state->nodes.push_back(std::move(n));
  }

  // ---- Per-kind params (typed projections of the verbatim params) --------
  for (auto &n : state->nodes) {
    const std::string ctx = "node '" + n.id + "'";
    switch (n.kind) {
    case node_kind::d_apply: {
      reject_unknown_keys(n.params,
                          {"num_measurements", "num_detectors", "d_sparse"},
                          ctx + " params");
      n.num_measurements = size_param(n.params, "num_measurements", ctx);
      auto num_detectors = size_param(n.params, "num_detectors", ctx);
      n.d_rows =
          nested_index_list(require_key(n.params, "d_sparse", ctx + " params"),
                            num_detectors, n.num_measurements,
                            ctx + " d_sparse");
      break;
    }
    case node_kind::decode: {
      reject_unknown_keys(
          n.params,
          {"detector_domain", "num_mechanisms", "h_sparse", "priors"},
          ctx + " params");
      n.num_mechanisms = size_param(n.params, "num_mechanisms", ctx);
      if (n.binding_ref.empty())
        fail(ctx + " is missing required key 'binding_ref'");
      if (n.binding_ref.rfind(decoder_binding_prefix, 0) != 0 ||
          n.binding_ref.size() == decoder_binding_prefix.size())
        fail(ctx + " binding_ref '" + n.binding_ref +
             "' is not of the form 'decoder:<plugin_name>'");
      break;
    }
    case node_kind::o_project: {
      reject_unknown_keys(n.params, {"num_observables", "o_sparse"},
                          ctx + " params");
      n.num_observables = size_param(n.params, "num_observables", ctx);
      if (n.output_ports.size() != n.num_observables)
        fail(ctx + " must declare one output port per observable (has " +
             std::to_string(n.output_ports.size()) + ", num_observables=" +
             std::to_string(n.num_observables) + ")");
      for (std::size_t k = 0; k < n.output_ports.size(); ++k)
        if (n.output_ports[k] != "l" + std::to_string(k))
          fail(ctx + " output port " + std::to_string(k) + " must be named 'l" +
               std::to_string(k) + "', got '" + n.output_ports[k] + "'");
      n.o_rows =
          nested_index_list(require_key(n.params, "o_sparse", ctx + " params"),
                            n.num_observables,
                            std::numeric_limits<std::uint32_t>::max(),
                            ctx + " o_sparse");
      break;
    }
    case node_kind::gf2_xor:
      reject_unknown_keys(n.params, {}, ctx + " params");
      break;
    case node_kind::root: {
      reject_unknown_keys(n.params, {"observable_index"}, ctx + " params");
      n.observable_index = size_param(n.params, "observable_index", ctx);
      break;
    }
    }
  }

  // ---- Edges --------------------------------------------------------------
  const auto &edges_j = require_key(doc, "edges", "IR top level");
  if (!edges_j.is_array())
    fail("'edges' must be an array");
  auto port_declared = [](const std::vector<std::string> &ports,
                          const std::string &p) {
    return std::find(ports.begin(), ports.end(), p) != ports.end();
  };
  for (const auto &ej : edges_j) {
    auto parts = string_list(ej, "edge");
    if (parts.size() != 4)
      fail("every edge must be [src_node, src_port, dst_node, dst_port]");
    edge e{parts[0], parts[1], parts[2], parts[3]};
    auto src_it = state->node_index.find(e.src_node);
    auto dst_it = state->node_index.find(e.dst_node);
    if (src_it == state->node_index.end())
      fail("edge references unknown source node '" + e.src_node + "'");
    if (dst_it == state->node_index.end())
      fail("edge references unknown destination node '" + e.dst_node + "'");
    const auto &src = state->nodes[src_it->second];
    auto &dst = state->nodes[dst_it->second];
    if (!port_declared(src.output_ports, e.src_port))
      fail("edge source port '" + e.src_node + "." + e.src_port +
           "' is not a declared output");
    if (!port_declared(dst.input_ports, e.dst_port))
      fail("edge destination port '" + e.dst_node + "." + e.dst_port +
           "' is not a declared input");
    if (output_type(src.kind) != input_type(dst.kind))
      fail("edge '" + e.src_node + "." + e.src_port + "' -> '" + e.dst_node +
           "." + e.dst_port + "' connects incompatible value types");
    // Record the feed.
    dst.feeds.resize(dst.input_ports.size(), {-2, ""});
    for (std::size_t i = 0; i < dst.input_ports.size(); ++i)
      if (dst.input_ports[i] == e.dst_port) {
        if (dst.feeds[i].first != -2)
          fail("input port '" + e.dst_node + "." + e.dst_port +
               "' is fed more than once");
        dst.feeds[i] = {static_cast<int>(src_it->second), e.src_port};
      }
    state->edges.push_back(std::move(e));
  }

  // ---- Graph inputs ---------------------------------------------------
  const auto &inputs_j = require_key(doc, "inputs", "IR top level");
  if (!inputs_j.is_object() || inputs_j.empty())
    fail("'inputs' must be a non-empty object");
  for (const auto &item : inputs_j.items()) {
    auto target = string_list(item.value(), "graph input '" + item.key() + "'");
    if (target.size() != 2)
      fail("graph input '" + item.key() + "' must be [node, port]");
    auto it = state->node_index.find(target[0]);
    if (it == state->node_index.end())
      fail("graph input '" + item.key() + "' references unknown node '" +
           target[0] + "'");
    auto &n = state->nodes[it->second];
    if (input_type(n.kind) != port_type::measurements)
      fail("graph input '" + item.key() +
           "' must feed a measurement-consuming node (d_apply)");
    n.feeds.resize(n.input_ports.size(), {-2, ""});
    bool found = false;
    for (std::size_t i = 0; i < n.input_ports.size(); ++i)
      if (n.input_ports[i] == target[1]) {
        if (n.feeds[i].first != -2)
          fail("input port '" + target[0] + "." + target[1] +
               "' is fed more than once");
        n.feeds[i] = {-1, ""};
        found = true;
      }
    if (!found)
      fail("graph input '" + item.key() + "' references undeclared port '" +
           target[0] + "." + target[1] + "'");
    state->graph_inputs.emplace_back(item.key(), target[0], target[1]);
  }

  // Every declared input port must be fed.
  for (auto &n : state->nodes) {
    n.feeds.resize(n.input_ports.size(), {-2, ""});
    for (std::size_t i = 0; i < n.input_ports.size(); ++i)
      if (n.feeds[i].first == -2)
        fail("input port '" + n.id + "." + n.input_ports[i] +
             "' has no incoming edge or graph input");
  }

  // ---- Roots ----------------------------------------------------------
  const auto &roots_j = require_key(doc, "roots", "IR top level");
  state->root_ids = string_list(roots_j, "'roots'");
  if (state->root_ids.empty())
    fail("'roots' must be a non-empty array");
  std::map<std::size_t, std::size_t> by_observable;
  std::unordered_set<std::string> listed;
  for (const auto &rid : state->root_ids) {
    auto it = state->node_index.find(rid);
    if (it == state->node_index.end())
      fail("'roots' references unknown node '" + rid + "'");
    const auto &n = state->nodes[it->second];
    if (n.kind != node_kind::root)
      fail("'roots' entry '" + rid + "' is not a root-kind node");
    if (!listed.insert(rid).second)
      fail("'roots' lists node '" + rid + "' more than once");
    if (!by_observable.emplace(n.observable_index, it->second).second)
      fail("duplicate root observable_index " +
           std::to_string(n.observable_index));
  }
  for (const auto &n : state->nodes)
    if (n.kind == node_kind::root && !listed.count(n.id))
      fail("root-kind node '" + n.id + "' is missing from 'roots'");
  for (const auto &[obs, idx] : by_observable) {
    state->roots_by_observable.push_back(idx);
    state->output_names.push_back(state->nodes[idx].id);
  }

  // ---- Topological order (Kahn) ----------------------------------------
  {
    std::vector<std::size_t> indegree(state->nodes.size(), 0);
    std::vector<std::vector<std::size_t>> successors(state->nodes.size());
    for (std::size_t i = 0; i < state->nodes.size(); ++i)
      for (const auto &[producer, port] : state->nodes[i].feeds)
        if (producer >= 0) {
          ++indegree[i];
          successors[static_cast<std::size_t>(producer)].push_back(i);
        }
    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < indegree.size(); ++i)
      if (indegree[i] == 0)
        ready.push_back(i);
    while (!ready.empty()) {
      auto i = ready.front();
      ready.pop_front();
      state->topo_order.push_back(i);
      for (auto s : successors[i])
        if (--indegree[s] == 0)
          ready.push_back(s);
    }
    if (state->topo_order.size() != state->nodes.size())
      fail("graph contains a cycle");
  }

  // ---- Construct decoders (D/H/O and decoders fixed at construction) ----
  for (auto &n : state->nodes) {
    if (n.kind != node_kind::decode)
      continue;
    const std::string ctx = "node '" + n.id + "'";

    // The number of detectors this decoder sees: the detector_domain size
    // when given, otherwise the full detector count of the producing
    // d_apply node.
    std::size_t producer_detectors = 0;
    if (n.feeds[0].first >= 0) {
      const auto &src = state->nodes[static_cast<std::size_t>(n.feeds[0].first)];
      if (src.kind == node_kind::d_apply)
        producer_detectors = src.d_rows.size();
    }
    std::size_t num_detectors = producer_detectors;
    if (auto it = n.params.find("detector_domain"); it != n.params.end()) {
      auto domain_rows = nested_index_list(
          json::array({*it}), 1,
          producer_detectors ? producer_detectors
                             : std::numeric_limits<std::uint32_t>::max(),
          ctx + " detector_domain");
      n.detector_domain = std::move(domain_rows[0]);
      num_detectors = n.detector_domain.size();
    } else if (producer_detectors == 0)
      fail(ctx + " needs 'detector_domain' (its detector count cannot be "
                 "inferred from its producer)");

    auto h_rows =
        nested_index_list(require_key(n.params, "h_sparse", ctx + " params"),
                          n.num_mechanisms, num_detectors, ctx + " h_sparse");
    auto H = sparse_binary_matrix::from_nested_csc(
        static_cast<std::uint32_t>(num_detectors),
        static_cast<std::uint32_t>(n.num_mechanisms), h_rows);

    const auto &priors_j = require_key(n.params, "priors", ctx + " params");
    if (!priors_j.is_array() || priors_j.size() != n.num_mechanisms)
      fail(ctx + " 'priors' must be an array of length num_mechanisms");
    std::vector<double> priors;
    priors.reserve(priors_j.size());
    for (const auto &p : priors_j) {
      if (!p.is_number())
        fail(ctx + " 'priors' contains a non-numeric entry");
      priors.push_back(p.get<double>());
    }

    auto plugin_name = n.binding_ref.substr(decoder_binding_prefix.size());
    decoder_init init(std::move(H), std::nullopt, std::move(priors));
    try {
      n.dec = get_decoder(plugin_name, std::move(init));
    } catch (const std::exception &e) {
      fail(ctx + ": failed to construct decoder '" + plugin_name +
           "': " + e.what());
    }
  }

  return decoding_task_graph(std::move(state));
}

// ---- Tasking-bundle loading ------------------------------------------------

namespace {

std::string read_file_bytes(const std::filesystem::path &path,
                            const std::string &ctx) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    fail(ctx + " cannot be read: " + path.string());
  std::string bytes((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  return bytes;
}

bool is_sha256_hex(const std::string &s) {
  if (s.size() != 64)
    return false;
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  return true;
}

/// Resolve a manifest/program URI strictly inside the bundle root.
std::filesystem::path resolve_inside(const std::filesystem::path &root,
                                     const std::string &uri) {
  std::filesystem::path relative(uri);
  if (relative.is_absolute())
    fail("bundle URI leaves its root: '" + uri + "'");
  for (const auto &part : relative)
    if (part == "..")
      fail("bundle URI leaves its root: '" + uri + "'");
  return root / relative;
}

std::string string_member(const json &obj, const char *key,
                          const std::string &ctx) {
  const auto &v = require_key(obj, key, ctx);
  if (!v.is_string())
    fail(ctx + " key '" + std::string(key) + "' must be a string");
  return v.get<std::string>();
}

/// Flat list of non-negative integers, bounded.
std::vector<std::uint32_t> index_list(const json &arr, std::size_t bound,
                                      const std::string &ctx) {
  auto rows = nested_index_list(json::array({arr}), 1, bound, ctx);
  return std::move(rows[0]);
}

/// One artifact row of the bundle manifest.
struct bundle_artifact {
  std::string uri;
  std::string sha256;
  std::uint64_t size_bytes = 0;
};

/// The parsed, binding-checked project-to-commit payload of one view.
struct project_view_payload {
  std::size_t candidate_count = 0;
  json candidate_binding;
  json logical_binding;
  json detector_binding;
  json responsibility_scope;
  bool logical_from_solve = false;
  std::vector<std::vector<std::uint32_t>> logical_rows;
  std::vector<std::vector<std::uint32_t>> detector_rows;
};

project_view_payload parse_project_view(const std::string &payload,
                                        const std::string &view_id) {
  const std::string ctx = "project view '" + view_id + "'";
  json doc;
  try {
    doc = json::parse(payload);
  } catch (const json::parse_error &e) {
    fail(ctx + " payload is not valid JSON: " + e.what());
  }
  if (!doc.is_object())
    fail(ctx + " payload must be a JSON object");
  if (string_member(doc, "schema_version", ctx) != project_view_schema_version)
    fail(ctx + " has an unsupported project-to-commit schema");

  project_view_payload view;
  const auto &count = require_key(doc, "candidate_count", ctx);
  if (!count.is_number_unsigned() || count.get<std::uint64_t>() == 0)
    fail(ctx + " candidate_count must be a positive integer");
  view.candidate_count = count.get<std::size_t>();
  view.candidate_binding = require_key(doc, "candidate_binding", ctx);
  view.logical_binding = require_key(doc, "logical_binding", ctx);
  view.detector_binding = require_key(doc, "detector_binding", ctx);
  view.responsibility_scope = require_key(doc, "responsibility_scope", ctx);
  if (auto it = doc.find("logical_from_solve"); it != doc.end())
    view.logical_from_solve = it->is_boolean() && it->get<bool>();
  if (!view.logical_binding.is_array() || !view.detector_binding.is_array())
    fail(ctx + " bindings must be arrays");

  const std::size_t logical_count =
      view.logical_from_solve ? 0 : view.logical_binding.size();
  view.logical_rows =
      nested_index_list(require_key(doc, "logical_effect_rows", ctx),
                        logical_count, view.candidate_count,
                        ctx + " logical_effect_rows");
  view.detector_rows =
      nested_index_list(require_key(doc, "detector_effect_rows", ctx),
                        view.detector_binding.size(), view.candidate_count,
                        ctx + " detector_effect_rows");

  auto committed = index_list(require_key(doc, "committed_candidate_indices",
                                          ctx),
                              view.candidate_count,
                              ctx + " committed_candidate_indices");
  std::unordered_set<std::uint32_t> commit_mask(committed.begin(),
                                                committed.end());
  if (commit_mask.size() != committed.size())
    fail(ctx + " commit mask contains duplicates");
  for (const auto &rows : {view.logical_rows, view.detector_rows})
    for (const auto &row : rows)
      for (auto candidate : row)
        if (!commit_mask.count(candidate))
          fail(ctx + " effect maps contain candidates outside the commit "
                     "mask");
  return view;
}

/// Declared port from the program IR: wire-type name plus opaque binding.
struct declared_port {
  std::string type_name;
  json binding;
};

declared_port parse_port(const json &port_j, const std::string &ctx) {
  if (!port_j.is_object())
    fail(ctx + " ports must be objects");
  const auto &type_j = require_key(port_j, "type", ctx);
  declared_port port;
  port.type_name = string_member(type_j, "type", ctx + " port type");
  port.binding = type_j.contains("binding") ? type_j["binding"] : json::array();
  return port;
}

} // namespace

decoding_task_graph
decoding_task_graph::from_bundle(const std::string &bundle_dir,
                                 const std::string &decoder_name) {
  namespace fs = std::filesystem;
  const fs::path root(bundle_dir);

  // ---- Manifest: inventory + integrity ------------------------------------
  auto manifest_bytes = read_file_bytes(root / "manifest.json",
                                        "bundle manifest");
  json manifest;
  try {
    manifest = json::parse(manifest_bytes);
  } catch (const json::parse_error &e) {
    fail(std::string("bundle manifest is not valid JSON: ") + e.what());
  }
  if (!manifest.is_object())
    fail("bundle manifest must be a JSON object");
  if (string_member(manifest, "schema_version", "bundle manifest") !=
      bundle_schema_version)
    fail("unsupported bundle schema (expected '" +
         std::string(bundle_schema_version) + "')");

  const auto &program_ref = require_key(manifest, "program", "bundle manifest");
  auto program_uri = string_member(program_ref, "uri", "bundle program");
  auto program_sha = string_member(program_ref, "sha256", "bundle program");
  if (!is_sha256_hex(program_sha))
    fail("bundle program SHA-256 is malformed");
  auto program_bytes = read_file_bytes(resolve_inside(root, program_uri),
                                       "bundle program");
  if (sha256_hex(program_bytes) != program_sha)
    fail("bundle program SHA-256 verification failed");

  // (kind, artifact_id) -> artifact. Every inventoried artifact — including
  // audit sidecars the executor never consumes — is hash-verified.
  std::map<std::pair<std::string, std::string>, bundle_artifact> artifacts;
  const auto &artifacts_j = require_key(manifest, "artifacts",
                                        "bundle manifest");
  if (!artifacts_j.is_array())
    fail("bundle manifest 'artifacts' must be an array");
  for (const auto &aj : artifacts_j) {
    if (!aj.is_object())
      fail("bundle manifest artifacts must be objects");
    auto id = string_member(aj, "artifact_id", "bundle artifact");
    auto kind = string_member(aj, "artifact_kind", "bundle artifact");
    if (kind != "dem" && kind != "project_view" && kind != "audit")
      fail("unknown bundle artifact kind '" + kind + "'");
    bundle_artifact artifact;
    artifact.uri = string_member(aj, "uri", "bundle artifact");
    artifact.sha256 = string_member(aj, "sha256", "bundle artifact");
    if (!is_sha256_hex(artifact.sha256))
      fail("bundle artifact SHA-256 is malformed: " + artifact.uri);
    const auto &size_j = require_key(aj, "size_bytes", "bundle artifact");
    if (!size_j.is_number_unsigned())
      fail("bundle artifact size must be a non-negative integer");
    artifact.size_bytes = size_j.get<std::uint64_t>();
    auto payload = read_file_bytes(resolve_inside(root, artifact.uri),
                                   "bundle artifact");
    if (payload.size() != artifact.size_bytes)
      fail("bundle artifact size verification failed: " + artifact.uri);
    if (sha256_hex(payload) != artifact.sha256)
      fail("bundle artifact SHA-256 verification failed: " + artifact.uri);
    if (!artifacts.emplace(std::make_pair(kind, id), std::move(artifact))
             .second)
      fail("bundle artifact IDs are not unique by kind: " + kind + "/" + id);
  }

  // ---- Program envelope ----------------------------------------------------
  json program;
  try {
    program = json::parse(program_bytes);
  } catch (const json::parse_error &e) {
    fail(std::string("bundle program is not valid JSON: ") + e.what());
  }
  if (!program.is_object())
    fail("bundle program must be a JSON object");
  if (string_member(program, "schema_version", "bundle program") !=
      program_schema_version)
    fail("bundle program has an unsupported task-graph schema (expected '" +
         std::string(program_schema_version) + "')");

  // DEM references: cross-checked against the manifest inventory, payloads
  // kept for decoder construction.
  struct dem_reference {
    std::size_t num_detectors = 0;
    std::size_t num_observables = 0;
    std::string text;
  };
  std::unordered_map<std::string, dem_reference> dems;
  const auto &dems_j = require_key(program, "dems", "bundle program");
  if (!dems_j.is_array())
    fail("bundle program 'dems' must be an array");
  for (const auto &dj : dems_j) {
    auto id = string_member(dj, "id", "DEM reference");
    const std::string ctx = "DEM reference '" + id + "'";
    auto it = artifacts.find({"dem", id});
    if (it == artifacts.end())
      fail(ctx + " is missing from the bundle inventory");
    if (string_member(dj, "sha256", ctx) != it->second.sha256 ||
        string_member(dj, "uri", ctx) != it->second.uri)
      fail(ctx + " differs from the bundle inventory");
    dem_reference ref;
    const auto &nd = require_key(dj, "num_detectors", ctx);
    const auto &nobs = require_key(dj, "num_observables", ctx);
    if (!nd.is_number_unsigned() || !nobs.is_number_unsigned())
      fail(ctx + " detector/observable counts must be non-negative integers");
    ref.num_detectors = nd.get<std::size_t>();
    ref.num_observables = nobs.get<std::size_t>();
    ref.text = read_file_bytes(resolve_inside(root, it->second.uri), ctx);
    if (!dems.emplace(id, std::move(ref)).second)
      fail("duplicate DEM reference '" + id + "'");
  }

  // Correction-view references: cross-checked, payloads parsed, and the
  // payload bindings verified against the program's reference.
  std::unordered_map<std::string, project_view_payload> views;
  const auto &views_j = require_key(program, "correction_views",
                                    "bundle program");
  if (!views_j.is_array())
    fail("bundle program 'correction_views' must be an array");
  for (const auto &vj : views_j) {
    auto id = string_member(vj, "id", "correction-view reference");
    const std::string ctx = "correction-view reference '" + id + "'";
    auto it = artifacts.find({"project_view", id});
    if (it == artifacts.end())
      fail(ctx + " is missing from the bundle inventory");
    if (string_member(vj, "sha256", ctx) != it->second.sha256 ||
        string_member(vj, "uri", ctx) != it->second.uri)
      fail(ctx + " differs from the bundle inventory");
    auto view = parse_project_view(
        read_file_bytes(resolve_inside(root, it->second.uri), ctx), id);
    if (require_key(vj, "candidate_binding", ctx) != view.candidate_binding ||
        require_key(vj, "logical_binding", ctx) != view.logical_binding ||
        require_key(vj, "detector_binding", ctx) != view.detector_binding ||
        require_key(vj, "responsibility_scope", ctx) !=
            view.responsibility_scope)
      fail(ctx + " bindings differ from its payload");
    if (!views.emplace(id, std::move(view)).second)
      fail("duplicate correction-view reference '" + id + "'");
  }

  // Runtime inventory must cover the program's references exactly (audit
  // sidecars excepted).
  for (const auto &[key, artifact] : artifacts) {
    if (key.first == "dem" && !dems.count(key.second))
      fail("bundle inventories DEM '" + key.second +
           "' that the program does not reference");
    if (key.first == "project_view" && !views.count(key.second))
      fail("bundle inventories project view '" + key.second +
           "' that the program does not reference");
  }

  // ---- Tasks ----------------------------------------------------------------
  auto state = std::make_shared<impl>();
  state->loaded_from_bundle = true;
  state->graph_input_kind = impl::input_kind::detections;
  if (auto it = program.find("metadata"); it != program.end()) {
    state->has_metadata = true;
    state->metadata = *it;
  }

  // Declared type (name + binding) per node input/output port, for
  // data-driven edge type checking.
  std::map<std::pair<std::string, std::string>, declared_port> in_ports,
      out_ports;

  const auto &tasks_j = require_key(program, "tasks", "bundle program");
  if (!tasks_j.is_array() || tasks_j.empty())
    fail("bundle program 'tasks' must be a non-empty array");
  for (const auto &tj : tasks_j) {
    if (!tj.is_object())
      fail("every task must be a JSON object");
    node n;
    n.id = string_member(tj, "id", "task");
    const std::string ctx = "task '" + n.id + "'";
    if (tj.contains("body") && !tj["body"].is_null())
      fail(ctx + ": composite task bodies are not supported");
    if (tj.contains("guard") && !tj["guard"].is_null())
      fail(ctx + ": task guards are not supported");

    const std::string binding_ref = string_member(tj, "binding_ref", ctx);
    if (binding_ref == "ingest")
      n.kind = node_kind::ingest;
    else if (binding_ref == "compiled_decode")
      n.kind = node_kind::compiled_decode;
    else if (binding_ref == "compiled_effect_view")
      n.kind = node_kind::compiled_effect_view;
    else if (binding_ref == "compiled_contribution")
      n.kind = node_kind::compiled_contribution;
    else if (binding_ref == "combine_xor" || binding_ref == "xor")
      n.kind = node_kind::gf2_xor;
    else
      fail(ctx + " has unsupported binding_ref '" + binding_ref + "'");
    n.binding_ref = binding_ref;

    std::vector<declared_port> input_types, output_types;
    for (const auto &pj : require_key(tj, "inputs", ctx)) {
      auto name = string_member(pj, "name", ctx + " input");
      auto port = parse_port(pj, ctx + " input '" + name + "'");
      n.input_ports.push_back(name);
      input_types.push_back(port);
      if (!in_ports.emplace(std::make_pair(n.id, name), std::move(port))
               .second)
        fail(ctx + " declares input port '" + name + "' twice");
    }
    for (const auto &pj : require_key(tj, "outputs", ctx)) {
      auto name = string_member(pj, "name", ctx + " output");
      auto port = parse_port(pj, ctx + " output '" + name + "'");
      n.output_ports.push_back(name);
      output_types.push_back(port);
      if (!out_ports.emplace(std::make_pair(n.id, name), std::move(port))
               .second)
        fail(ctx + " declares output port '" + name + "' twice");
    }

    switch (n.kind) {
    case node_kind::ingest:
      if (n.input_ports.size() != 1 || n.output_ports.size() != 1 ||
          input_types[0].type_name != output_types[0].type_name)
        fail(ctx + ": ingest must forward exactly one port unchanged");
      break;

    case node_kind::compiled_decode:
    case node_kind::compiled_contribution: {
      // Detector domain: the global detector indices this solve sees, in
      // declared order (order defines the local detector index).
      const auto &partition = require_key(tj, "partition", ctx);
      if (!partition.is_object())
        fail(ctx + " 'partition' must be an object");
      n.compiled_domain = index_list(
          require_key(partition, "domain_detectors", ctx + " partition"),
          std::numeric_limits<std::uint32_t>::max(),
          ctx + " domain_detectors");
      std::unordered_map<std::uint32_t, std::uint32_t> local_of_global;
      for (std::size_t i = 0; i < n.compiled_domain.size(); ++i)
        if (!local_of_global
                 .emplace(n.compiled_domain[i],
                          static_cast<std::uint32_t>(i))
                 .second)
          fail(ctx + " domain_detectors contains duplicates");

      // Input ports: the full detection-event vector on 's', plus typed
      // detector-delta ports whose binding names the global detector index
      // of each delta bit.
      n.delta_positions.resize(n.input_ports.size());
      bool have_syndrome = false;
      for (std::size_t i = 0; i < n.input_ports.size(); ++i) {
        if (input_types[i].type_name != "detection_events")
          fail(ctx + " input '" + n.input_ports[i] +
               "' must carry detection_events");
        if (n.input_ports[i] == "s") {
          n.syndrome_slot = i;
          have_syndrome = true;
          continue;
        }
        auto bound = index_list(input_types[i].binding,
                                std::numeric_limits<std::uint32_t>::max(),
                                ctx + " input '" + n.input_ports[i] +
                                    "' binding");
        if (bound.empty())
          fail(ctx + " delta input '" + n.input_ports[i] +
               "' has an empty detector binding");
        for (auto detector : bound) {
          auto it = local_of_global.find(detector);
          if (it == local_of_global.end())
            fail(ctx + " delta input '" + n.input_ports[i] +
                 "' leaves the solve's detector domain");
          n.delta_positions[i].push_back(it->second);
        }
      }
      if (!have_syndrome)
        fail(ctx + " must declare a detection-event input port 's'");

      // The solve's own local DEM fixes decoder and candidate basis now.
      auto dem_id = string_member(tj, "dem_ref", ctx);
      auto dem_it = dems.find(dem_id);
      if (dem_it == dems.end())
        fail(ctx + " references unknown DEM '" + dem_id + "'");
      const auto &ref = dem_it->second;
      if (ref.num_detectors != n.compiled_domain.size())
        fail(ctx + " detector domain size " +
             std::to_string(n.compiled_domain.size()) +
             " differs from its DEM's " + std::to_string(ref.num_detectors) +
             " detectors");
      // Decomposition suggestions expanded: one column per graphlike
      // component, matching the reference runtime's
      // pymatching.Matching.from_detector_error_model handling.
      auto model = dem_from_stim_text(ref.text,
                                      /*use_decomp_suggestions=*/true);
      if (model.num_detectors() != ref.num_detectors ||
          model.num_observables() != ref.num_observables)
        fail(ctx + ": DEM '" + dem_id +
             "' counts differ from its program reference");
      n.candidate_count = ref.num_observables;
      try {
        n.dec = get_decoder(decoder_name, decoder_init(std::move(model)),
                            decode_result_type::observables);
      } catch (const std::exception &e) {
        fail(ctx + ": failed to construct decoder '" + decoder_name +
             "': " + e.what());
      }

      if (n.kind == node_kind::compiled_decode) {
        if (n.output_ports != std::vector<std::string>{"candidate"})
          fail(ctx + " must output exactly 'candidate' (a separate solve "
                     "logical output is not supported, matching the "
                     "reference binder)");
      }
      break;
    }

    case node_kind::compiled_effect_view:
      if (n.input_ports != std::vector<std::string>{"candidate"})
        fail(ctx + " must consume exactly 'candidate' (logical-from-solve "
                   "views are not supported, matching the reference binder)");
      break;

    case node_kind::gf2_xor: {
      const std::size_t min_inputs = binding_ref == "xor" ? 2 : 1;
      if (n.input_ports.size() < min_inputs || n.output_ports.size() != 1)
        fail(ctx + " must combine at least " + std::to_string(min_inputs) +
             " logical input(s) into one output");
      for (const auto &port : input_types)
        if (port.type_name != "logical_outcome")
          fail(ctx + " inputs must carry logical_outcome values");
      break;
    }

    default:
      break;
    }

    // Project-to-commit payload for view-owning kinds.
    if (n.kind == node_kind::compiled_effect_view ||
        n.kind == node_kind::compiled_contribution) {
      auto view_id = string_member(tj, "correction_view_ref", ctx);
      auto view_it = views.find(view_id);
      if (view_it == views.end())
        fail(ctx + " references unknown correction view '" + view_id + "'");
      const auto &view = view_it->second;
      if (view.logical_from_solve)
        fail(ctx + ": logical-from-solve views are not supported, matching "
                   "the reference binder");
      n.logical_rows = view.logical_rows;
      n.detector_rows = view.detector_rows;
      n.has_delta_output = !view.detector_rows.empty();
      if (n.kind == node_kind::compiled_effect_view)
        n.candidate_count = view.candidate_count;
      else if (view.candidate_count != n.candidate_count)
        fail(ctx + ": correction view '" + view_id +
             "' candidate basis differs from the solve's DEM");
      const std::vector<std::string> want_outputs =
          n.has_delta_output ? std::vector<std::string>{"L", "delta"}
                             : std::vector<std::string>{"L"};
      if (n.output_ports != want_outputs)
        fail(ctx + " outputs must be exactly 'L'" +
             (n.has_delta_output ? " and 'delta'" : "") +
             " for its correction view");
    }

    if (!state->node_index.emplace(n.id, state->nodes.size()).second)
      fail("duplicate task id '" + n.id + "'");
    state->nodes.push_back(std::move(n));
  }

  // ---- Edges (typed by the IR's declared port types) -----------------------
  for (const auto &ej : require_key(program, "edges", "bundle program")) {
    if (!ej.is_array() || ej.size() != 2)
      fail("every edge must be [[src_task, src_port], [dst_task, dst_port]]");
    auto src = string_list(ej[0], "edge endpoint");
    auto dst = string_list(ej[1], "edge endpoint");
    if (src.size() != 2 || dst.size() != 2)
      fail("every edge endpoint must be [task, port]");
    edge e{src[0], src[1], dst[0], dst[1]};
    auto src_it = state->node_index.find(e.src_node);
    auto dst_it = state->node_index.find(e.dst_node);
    if (src_it == state->node_index.end())
      fail("edge references unknown source task '" + e.src_node + "'");
    if (dst_it == state->node_index.end())
      fail("edge references unknown destination task '" + e.dst_node + "'");
    auto src_port = out_ports.find({e.src_node, e.src_port});
    auto dst_port = in_ports.find({e.dst_node, e.dst_port});
    if (src_port == out_ports.end())
      fail("edge source port '" + e.src_node + "." + e.src_port +
           "' is not a declared output");
    if (dst_port == in_ports.end())
      fail("edge destination port '" + e.dst_node + "." + e.dst_port +
           "' is not a declared input");
    if (src_port->second.type_name != dst_port->second.type_name ||
        src_port->second.binding != dst_port->second.binding)
      fail("edge '" + e.src_node + "." + e.src_port + "' -> '" + e.dst_node +
           "." + e.dst_port + "' connects incompatible port types");
    auto &dst_node = state->nodes[dst_it->second];
    dst_node.feeds.resize(dst_node.input_ports.size(), {-2, ""});
    for (std::size_t i = 0; i < dst_node.input_ports.size(); ++i)
      if (dst_node.input_ports[i] == e.dst_port) {
        if (dst_node.feeds[i].first != -2)
          fail("input port '" + e.dst_node + "." + e.dst_port +
               "' is fed more than once");
        dst_node.feeds[i] = {static_cast<int>(src_it->second), e.src_port};
      }
    state->edges.push_back(std::move(e));
  }

  // Every effect view's candidate producer must share its candidate basis
  // width (binding equality was already enforced on the edge).
  for (auto &n : state->nodes) {
    if (n.kind != node_kind::compiled_effect_view)
      continue;
    n.feeds.resize(n.input_ports.size(), {-2, ""});
    if (n.feeds[0].first < 0)
      fail("effect view '" + n.id + "' has no candidate producer");
    const auto &producer = state->nodes[static_cast<std::size_t>(
        n.feeds[0].first)];
    if (producer.kind != node_kind::compiled_decode)
      fail("effect view '" + n.id + "' must consume a compiled_decode "
                                    "candidate");
    if (producer.candidate_count != n.candidate_count)
      fail("effect view '" + n.id + "' candidate basis width " +
           std::to_string(n.candidate_count) + " differs from solve '" +
           producer.id + "' width " +
           std::to_string(producer.candidate_count));
  }

  // ---- External inputs ------------------------------------------------------
  const auto &inputs_j = require_key(program, "external_inputs",
                                     "bundle program");
  if (!inputs_j.is_object() || inputs_j.size() != 1)
    fail("bundle program must declare exactly one external input");
  for (const auto &item : inputs_j.items()) {
    auto target = string_list(item.value(),
                              "external input '" + item.key() + "'");
    if (target.size() != 2)
      fail("external input '" + item.key() + "' must be [task, port]");
    auto it = state->node_index.find(target[0]);
    if (it == state->node_index.end())
      fail("external input '" + item.key() + "' references unknown task '" +
           target[0] + "'");
    auto port_it = in_ports.find({target[0], target[1]});
    if (port_it == in_ports.end())
      fail("external input '" + item.key() +
           "' references undeclared port '" + target[0] + "." + target[1] +
           "'");
    if (port_it->second.type_name != "detection_events")
      fail("external input '" + item.key() +
           "' must carry detection_events (measurement-to-detector "
           "conversion happens upstream of these bundles)");
    auto &n = state->nodes[it->second];
    n.feeds.resize(n.input_ports.size(), {-2, ""});
    for (std::size_t i = 0; i < n.input_ports.size(); ++i)
      if (n.input_ports[i] == target[1]) {
        if (n.feeds[i].first != -2)
          fail("input port '" + target[0] + "." + target[1] +
               "' is fed more than once");
        n.feeds[i] = {-1, ""};
      }
    state->graph_inputs.emplace_back(item.key(), target[0], target[1]);
  }

  // ---- External outputs become roots ---------------------------------------
  // nlohmann JSON objects iterate in lexicographic key order, so the root
  // order (and output_names()) is the external output names sorted.
  const auto &outputs_j = require_key(program, "external_outputs",
                                      "bundle program");
  if (!outputs_j.is_object() || outputs_j.empty())
    fail("bundle program must declare at least one external output");
  for (const auto &item : outputs_j.items()) {
    auto target = string_list(item.value(),
                              "external output '" + item.key() + "'");
    if (target.size() != 2)
      fail("external output '" + item.key() + "' must be [task, port]");
    auto src_it = state->node_index.find(target[0]);
    if (src_it == state->node_index.end())
      fail("external output '" + item.key() + "' references unknown task '" +
           target[0] + "'");
    auto port_it = out_ports.find({target[0], target[1]});
    if (port_it == out_ports.end())
      fail("external output '" + item.key() +
           "' references undeclared port '" + target[0] + "." + target[1] +
           "'");
    if (port_it->second.type_name != "logical_outcome")
      fail("external output '" + item.key() +
           "' must carry a logical_outcome");
    node root;
    root.id = "__root::" + item.key();
    root.kind = node_kind::root;
    root.observable_index = state->roots_by_observable.size();
    root.input_ports = {"L"};
    root.feeds = {{static_cast<int>(src_it->second), target[1]}};
    if (!state->node_index.emplace(root.id, state->nodes.size()).second)
      fail("external output '" + item.key() + "' collides with task id '" +
           root.id + "'");
    state->root_ids.push_back(root.id);
    state->roots_by_observable.push_back(state->nodes.size());
    state->output_names.push_back(item.key());
    state->nodes.push_back(std::move(root));
  }

  // ---- Wiring completeness + topological order -----------------------------
  for (auto &n : state->nodes) {
    n.feeds.resize(n.input_ports.size(), {-2, ""});
    for (std::size_t i = 0; i < n.input_ports.size(); ++i)
      if (n.feeds[i].first == -2)
        fail("input port '" + n.id + "." + n.input_ports[i] +
             "' has no incoming edge or external input");
  }
  {
    std::vector<std::size_t> indegree(state->nodes.size(), 0);
    std::vector<std::vector<std::size_t>> successors(state->nodes.size());
    for (std::size_t i = 0; i < state->nodes.size(); ++i)
      for (const auto &[producer, port] : state->nodes[i].feeds)
        if (producer >= 0) {
          ++indegree[i];
          successors[static_cast<std::size_t>(producer)].push_back(i);
        }
    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < indegree.size(); ++i)
      if (indegree[i] == 0)
        ready.push_back(i);
    while (!ready.empty()) {
      auto i = ready.front();
      ready.pop_front();
      state->topo_order.push_back(i);
      for (auto s : successors[i])
        if (--indegree[s] == 0)
          ready.push_back(s);
    }
    if (state->topo_order.size() != state->nodes.size())
      fail("graph contains a cycle");
  }

  return decoding_task_graph(std::move(state));
}

namespace {

/// Synchronous topological walk shared by both run() overloads. The graph
/// input value feeds every input slot whose recorded producer is -1.
/// (Takes the graph pieces rather than the pimpl struct: `impl` is a private
/// nested type this file-local function may not name.)
struct graph_pieces {
  std::vector<node> &nodes;
  const std::vector<std::size_t> &topo_order;
  const std::vector<std::size_t> &roots_by_observable;
};

std::vector<logical_outcome> execute_graph(graph_pieces g,
                                           const port_value &graph_input) {
  // Output values, per node per output port.
  std::vector<std::unordered_map<std::string, port_value>> outputs(
      g.nodes.size());
  std::vector<logical_outcome> root_values(g.nodes.size());

  auto input_of = [&](const node &n, std::size_t slot) -> const port_value & {
    const auto &[producer, port] = n.feeds[slot];
    if (producer < 0)
      return graph_input;
    return outputs[static_cast<std::size_t>(producer)].at(port);
  };

  // Emit the logical contribution (and detector delta) of a project-to-
  // commit payload applied to a hard candidate vector.
  auto apply_view = [&](const node &n, std::size_t idx,
                        const std::vector<std::uint8_t> &candidate,
                        bool converged, std::size_t logical_port,
                        std::size_t delta_port) {
    logical_outcome logical;
    logical.converged = converged;
    logical.bits.reserve(n.logical_rows.size());
    for (const auto &row : n.logical_rows) {
      std::uint8_t bit = 0;
      for (auto c : row)
        bit ^= candidate[c];
      logical.bits.push_back(bit);
    }
    outputs[idx].emplace(n.output_ports[logical_port], std::move(logical));
    if (!n.has_delta_output)
      return;
    detection_events delta;
    delta.converged = converged;
    delta.events.reserve(n.detector_rows.size());
    for (const auto &row : n.detector_rows) {
      std::uint8_t bit = 0;
      for (auto c : row)
        bit ^= candidate[c];
      delta.events.push_back(static_cast<float_t>(bit));
    }
    outputs[idx].emplace(n.output_ports[delta_port], std::move(delta));
  };

  for (auto idx : g.topo_order) {
    auto &n = g.nodes[idx];
    switch (n.kind) {
    case node_kind::d_apply: {
      const auto &measurements =
          std::get<measurement_results>(input_of(n, 0));
      if (measurements.bits.size() != n.num_measurements)
        fail("node '" + n.id + "' expects " +
             std::to_string(n.num_measurements) + " measurement bits, got " +
             std::to_string(measurements.bits.size()));
      detection_events out;
      out.events.reserve(n.d_rows.size());
      for (const auto &row : n.d_rows) {
        std::uint8_t bit = 0;
        for (auto m : row)
          bit ^= (measurements.bits[m] & 1u);
        out.events.push_back(static_cast<float_t>(bit));
      }
      outputs[idx].emplace(n.output_ports[0], std::move(out));
      break;
    }
    case node_kind::ingest:
      outputs[idx].emplace(n.output_ports[0],
                           std::get<detection_events>(input_of(n, 0)));
      break;
    case node_kind::compiled_decode:
    case node_kind::compiled_contribution: {
      // Slice this solve's declared detector domain out of the global
      // detection events, then XOR in every typed detector delta at the
      // precomputed local positions.
      const auto &in =
          std::get<detection_events>(input_of(n, n.syndrome_slot));
      bool converged = in.converged;
      std::vector<float_t> syndrome;
      syndrome.reserve(n.compiled_domain.size());
      for (auto d : n.compiled_domain) {
        if (d >= in.events.size())
          fail("node '" + n.id + "' domain detector " + std::to_string(d) +
               " is out of range for " + std::to_string(in.events.size()) +
               " detection events");
        syndrome.push_back(in.events[d]);
      }
      for (std::size_t slot = 0; slot < n.input_ports.size(); ++slot) {
        if (slot == n.syndrome_slot)
          continue;
        const auto &delta = std::get<detection_events>(input_of(n, slot));
        const auto &positions = n.delta_positions[slot];
        if (delta.events.size() != positions.size())
          fail("node '" + n.id + "' delta input '" + n.input_ports[slot] +
               "' has width " + std::to_string(delta.events.size()) +
               ", expected " + std::to_string(positions.size()));
        converged = converged && delta.converged;
        for (std::size_t i = 0; i < positions.size(); ++i) {
          const std::uint8_t bit =
              (convert_soft_to_hard(syndrome[positions[i]]) ? 1u : 0u) ^
              (convert_soft_to_hard(delta.events[i]) ? 1u : 0u);
          syndrome[positions[i]] = static_cast<float_t>(bit);
        }
      }
      auto res = n.dec->decode(syndrome);
      converged = converged && res.converged;
      if (res.result.size() != n.candidate_count)
        fail("node '" + n.id + "' produced " +
             std::to_string(res.result.size()) + " candidates, expected " +
             std::to_string(n.candidate_count));
      std::vector<std::uint8_t> candidate(n.candidate_count);
      for (std::size_t i = 0; i < n.candidate_count; ++i)
        candidate[i] = convert_soft_to_hard(res.result[i]) ? 1u : 0u;
      if (n.kind == node_kind::compiled_decode)
        outputs[idx].emplace(n.output_ports[0],
                             correction_candidate{std::move(candidate),
                                                  converged});
      else
        apply_view(n, idx, candidate, converged, /*logical_port=*/0,
                   /*delta_port=*/1);
      break;
    }
    case node_kind::compiled_effect_view: {
      const auto &in = std::get<correction_candidate>(input_of(n, 0));
      if (in.bits.size() != n.candidate_count)
        fail("node '" + n.id + "' received " +
             std::to_string(in.bits.size()) + " candidates, expected " +
             std::to_string(n.candidate_count));
      apply_view(n, idx, in.bits, in.converged, /*logical_port=*/0,
                 /*delta_port=*/1);
      break;
    }
    case node_kind::decode: {
      const auto &in = std::get<detection_events>(input_of(n, 0));
      std::vector<float_t> syndrome;
      if (!n.detector_domain.empty()) {
        syndrome.reserve(n.detector_domain.size());
        for (auto d : n.detector_domain) {
          if (d >= in.events.size())
            fail("node '" + n.id + "' detector_domain index " +
                 std::to_string(d) + " is out of range for " +
                 std::to_string(in.events.size()) + " detection events");
          syndrome.push_back(in.events[d]);
        }
      } else
        syndrome = in.events;
      if (syndrome.size() != n.dec->get_syndrome_size())
        fail("node '" + n.id + "' received " + std::to_string(syndrome.size()) +
             " detection events; its H has " +
             std::to_string(n.dec->get_syndrome_size()) + " detector rows");
      auto res = n.dec->decode(syndrome);
      error_pattern out;
      out.converged = res.converged;
      out.result = std::move(res.result);
      if (res.opt_results)
        out.opt_results = std::move(res.opt_results);
      outputs[idx].emplace(n.output_ports[0], std::move(out));
      break;
    }
    case node_kind::o_project: {
      const auto &in = std::get<error_pattern>(input_of(n, 0));
      for (std::size_t k = 0; k < n.num_observables; ++k) {
        logical_outcome out;
        out.converged = in.converged;
        std::uint8_t bit = 0;
        for (auto mech : n.o_rows[k]) {
          if (mech >= in.result.size())
            fail("node '" + n.id + "' o_sparse index " + std::to_string(mech) +
                 " is out of range for " + std::to_string(in.result.size()) +
                 " error mechanisms");
          bit ^= convert_soft_to_hard(in.result[mech]) ? 1u : 0u;
        }
        out.bits.push_back(bit);
        outputs[idx].emplace(n.output_ports[k], std::move(out));
      }
      break;
    }
    case node_kind::gf2_xor: {
      // Variadic GF(2) fold: `xor` (v0) always has two inputs,
      // `combine_xor` (bundle aggregation) has one or more.
      logical_outcome out = std::get<logical_outcome>(input_of(n, 0));
      for (std::size_t slot = 1; slot < n.input_ports.size(); ++slot) {
        const auto &next = std::get<logical_outcome>(input_of(n, slot));
        if (next.bits.size() != out.bits.size())
          fail("xor node '" + n.id + "' inputs have mismatched widths (" +
               std::to_string(out.bits.size()) + " vs " +
               std::to_string(next.bits.size()) + ")");
        out.converged = out.converged && next.converged;
        for (std::size_t i = 0; i < out.bits.size(); ++i)
          out.bits[i] ^= next.bits[i];
      }
      outputs[idx].emplace(n.output_ports[0], std::move(out));
      break;
    }
    case node_kind::root:
      root_values[idx] = std::get<logical_outcome>(input_of(n, 0));
      break;
    }
  }

  std::vector<logical_outcome> results;
  results.reserve(g.roots_by_observable.size());
  for (auto idx : g.roots_by_observable)
    results.push_back(std::move(root_values[idx]));
  return results;
}

} // namespace

std::vector<logical_outcome>
decoding_task_graph::run(const measurement_results &measurements) {
  if (impl_->graph_input_kind != impl::input_kind::measurements)
    fail("this graph's declared input is detection events; call "
         "run(detection_events)");
  return execute_graph({impl_->nodes, impl_->topo_order,
                        impl_->roots_by_observable},
                       port_value{measurements});
}

std::vector<logical_outcome>
decoding_task_graph::run(const detection_events &events) {
  if (impl_->graph_input_kind != impl::input_kind::detections)
    fail("this graph's declared input is raw measurements; call "
         "run(measurement_results)");
  return execute_graph({impl_->nodes, impl_->topo_order,
                        impl_->roots_by_observable},
                       port_value{events});
}

const std::vector<std::string> &decoding_task_graph::output_names() const {
  return impl_->output_names;
}

std::string decoding_task_graph::to_ir_json() const {
  const auto &g = *impl_;
  if (g.loaded_from_bundle)
    fail("graphs loaded from a tasking bundle do not re-emit IR; the bundle "
         "directory is the canonical artifact");
  json doc = json::object();
  doc["schema_version"] = std::string(schema_version_v0);
  if (g.has_derived_from)
    doc["derived_from"] = g.derived_from;
  if (g.has_metadata)
    doc["metadata"] = g.metadata;

  json nodes = json::array();
  for (const auto &n : g.nodes) {
    json nj = json::object();
    nj["id"] = n.id;
    switch (n.kind) {
    case node_kind::d_apply:
      nj["kind"] = "d_apply";
      break;
    case node_kind::decode:
      nj["kind"] = "decode";
      break;
    case node_kind::o_project:
      nj["kind"] = "o_project";
      break;
    case node_kind::gf2_xor:
      nj["kind"] = "xor";
      break;
    case node_kind::root:
      nj["kind"] = "root";
      break;
    }
    nj["inputs"] = n.input_ports;
    nj["outputs"] = n.output_ports;
    if (n.has_params)
      nj["params"] = n.params;
    if (n.kind == node_kind::decode)
      nj["binding_ref"] = n.binding_ref;
    nodes.push_back(std::move(nj));
  }
  doc["nodes"] = std::move(nodes);

  json edges = json::array();
  for (const auto &e : g.edges)
    edges.push_back(json::array({e.src_node, e.src_port, e.dst_node,
                                 e.dst_port}));
  doc["edges"] = std::move(edges);

  json inputs = json::object();
  for (const auto &[name, node_id, port] : g.graph_inputs)
    inputs[name] = json::array({node_id, port});
  doc["inputs"] = std::move(inputs);

  doc["roots"] = g.root_ids;
  return doc.dump();
}

} // namespace cudaq::qec
