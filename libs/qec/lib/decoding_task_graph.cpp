/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoding_task_graph.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <algorithm>
#include <deque>
#include <limits>
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

enum class node_kind { d_apply, decode, o_project, gf2_xor, root };

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

/// Value flowing on a port during synchronous execution.
using port_value = std::variant<measurement_results, detection_events,
                                error_pattern, logical_outcome>;

/// Static type of a value on a port, used to type-check edges at load time.
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
  }
  fail("unreachable node kind");
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
  case node_kind::root:
    break;
  }
  fail("root nodes have no outputs");
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
  for (const auto &[obs, idx] : by_observable)
    state->roots_by_observable.push_back(idx);

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

std::vector<logical_outcome>
decoding_task_graph::run(const measurement_results &measurements) {
  auto &g = *impl_;
  // Output values, per node per output port.
  std::vector<std::unordered_map<std::string, port_value>> outputs(
      g.nodes.size());
  std::vector<logical_outcome> root_values(g.nodes.size());

  auto input_of = [&](const node &n, std::size_t slot) -> const port_value & {
    static const port_value graph_input_slot{};
    const auto &[producer, port] = n.feeds[slot];
    if (producer < 0)
      return graph_input_slot; // never used; measurements handled separately
    return outputs[static_cast<std::size_t>(producer)].at(port);
  };

  for (auto idx : g.topo_order) {
    auto &n = g.nodes[idx];
    switch (n.kind) {
    case node_kind::d_apply: {
      if (n.feeds[0].first != -1)
        fail("d_apply node '" + n.id +
             "' must be fed by a graph input in this executor");
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
      const auto &a = std::get<logical_outcome>(input_of(n, 0));
      const auto &b = std::get<logical_outcome>(input_of(n, 1));
      if (a.bits.size() != b.bits.size())
        fail("xor node '" + n.id + "' inputs have mismatched widths (" +
             std::to_string(a.bits.size()) + " vs " +
             std::to_string(b.bits.size()) + ")");
      logical_outcome out;
      out.converged = a.converged && b.converged;
      out.bits.resize(a.bits.size());
      for (std::size_t i = 0; i < a.bits.size(); ++i)
        out.bits[i] = a.bits[i] ^ b.bits[i];
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

std::string decoding_task_graph::to_ir_json() const {
  const auto &g = *impl_;
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
