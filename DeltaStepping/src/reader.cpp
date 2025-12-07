/**
 * @file reader.cpp
 * @brief Implementation of graph reading utilities for GPU-accelerated SSSP algorithms.
 * @author Daniel Zhao
 * @date 2025-11-30
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdexcept>
#include <cctype>
#include <unordered_map>
#include <algorithm>

#include "reader.hpp"

/**
 * @brief Helper function to trim a string (remove leading and trailing whitespace).
 * @param s The string to trim.
 * @return The trimmed string.
 */
static inline std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        start++;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    return s.substr(start, end - start);
}

/**
 * @brief Reads a graph from a text file in edge list format.
 * The file should contain one edge per line in the format: source destination <weight>
 * source and destinations are not necessarily zero-indexed; the function will reindex them to be zero-indexed.
 * weight might not be present; if absent, all weights are assumed to be @param default_weight.
 * comments starting with `#` or `%` are ignored.
 * @param filename The path to the text file containing the graph.
 * @param directed Whether the graph is directed. If false, each edge is added in both directions.
 * @param default_weight The default weight to assign to edges if no weight is specified in the file.
 * @return The constructed Graph object.
 */
Graph read_from_txt(const std::string& filename, bool directed, Graph::weight_t default_weight) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file: " + filename);

    std::vector<std::pair<Graph::index_t, Graph::index_t>> raw_edges;
    std::vector<Graph::weight_t> weights;
    
    // Hash Map for Re-indexing: Raw ID -> 0-based ID
    std::unordered_map<int64_t, Graph::index_t> id_map;
    Graph::index_t next_id = 0;

    std::string line;
    while (std::getline(in, line)) {
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#' || line[first] == '%') continue;

        std::stringstream ss(line);
        int64_t u_raw, v_raw;
        double w = default_weight;

        if (!(ss >> u_raw >> v_raw)) continue;
        if (ss >> w) {} // Optional weight

        // Re-index on the fly
        if (id_map.find(u_raw) == id_map.end()) id_map[u_raw] = next_id++;
        if (id_map.find(v_raw) == id_map.end()) id_map[v_raw] = next_id++;

        Graph::index_t u = id_map[u_raw];
        Graph::index_t v = id_map[v_raw];

        raw_edges.push_back({u, v});
        weights.push_back((float)w);

        if (!directed) {
            raw_edges.push_back({v, u});
            weights.push_back((float)w);
        }
    }

    return Graph(next_id, raw_edges, weights);
}

/**
 * @brief Reads a graph from a .gr file in DIMACS format. 
 * The file should contain comments starting with a `c`. a problem line starting with `p` that includes the number of vertices and edges,
 * and the edges starting with `a` in the format: source destination weight.
 * source and destinations are 1-indexed in the file; the function will reindex them to be zero-indexed.
 * @param filename The path to the .gr file containing the graph.
 * @param directed Whether the graph is directed. If false, each edge is added in both directions.
 * @return The constructed Graph object.
 */
Graph read_from_gr(const std::string& filename, bool directed) {
    using index_t  = Graph::index_t;
    using weight_t = Graph::weight_t;

    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    index_t num_vertices = -1;
    long long num_edges_hint = -1;

    std::vector<std::pair<index_t, index_t>> edges;
    std::vector<weight_t> weights;
    edges.reserve(1024);
    weights.reserve(1024);

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;

        char c = line[0];
        if (c == 'c') {
            // comment line
            continue;
        }
        if (c == 'p') {
            // problem line: p sp <num_vertices> <num_edges>
            std::istringstream iss(line);
            char p_char;
            std::string sp_or_gr;
            long long n_nodes, n_edges;
            if (!(iss >> p_char >> sp_or_gr >> n_nodes >> n_edges)) {
                throw std::runtime_error("Malformed 'p' line in DIMACS file: " + filename);
            }
            if (n_nodes <= 0) {
                throw std::runtime_error("Non-positive number of nodes in DIMACS file: " + filename);
            }
            if (n_nodes > std::numeric_limits<index_t>::max()) {
                throw std::runtime_error("DIMACS graph too large for Graph::index_t");
            }
            num_vertices = static_cast<index_t>(n_nodes);
            num_edges_hint = n_edges;

            if (num_edges_hint > 0) {
                edges.reserve(static_cast<std::size_t>(num_edges_hint));
                weights.reserve(static_cast<std::size_t>(num_edges_hint));
            }
            continue;
        }
        if (c == 'a') {
            // arc line: a <u> <v> <w>
            std::istringstream iss(line);
            char a_char;
            long long u64, v64;
            double wtmp;
            if (!(iss >> a_char >> u64 >> v64 >> wtmp)) {
                throw std::runtime_error("Malformed 'a' line in DIMACS file: " + filename);
            }

            if (num_vertices < 0) {
                throw std::runtime_error("Found 'a' line before 'p' line in DIMACS file: " + filename);
            }

            // DIMACS nodes are 1-based; convert to 0-based
            long long u_zb = u64 - 1;
            long long v_zb = v64 - 1;

            if (u_zb < 0 || u_zb >= num_vertices || v_zb < 0 || v_zb >= num_vertices) {
                throw std::runtime_error("Node ID out of range in DIMACS file: " + filename);
            }

            index_t u = static_cast<index_t>(u_zb);
            index_t v = static_cast<index_t>(v_zb);
            weight_t w = static_cast<weight_t>(wtmp);

            edges.emplace_back(u, v);
            weights.push_back(w);

            if (!directed && u != v) {
                edges.emplace_back(v, u);
                weights.push_back(w);
            }
        }
        // Other line types are ignored.
    }

    if (num_vertices < 0) {
        throw std::runtime_error("No 'p' line found in DIMACS file: " + filename);
    }

    Graph g(num_vertices, edges, weights);
    return g;
}

/**
 * @brief Helper function to check if a string ends with a given suffix.
 * @param s The string to check.
 * @param suf The suffix to look for.
 * @return True if str ends with suffix, false otherwise.
 */
static bool ends_with(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    for (std::size_t i = 0; i < suf.size(); ++i) {
        char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(s[s.size() - suf.size() + i])));
        char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(suf[i])));
        if (c1 != c2) return false;
    }
    return true;
}

/**
 * @brief Reads a graph from a file
 * This would automatically detect the file format based on the file extension.
 * Supported formats are:
 * - .txt : edge list format
 * - .gr  : DIMACS format
 * @param filename The path to the file containing the graph.
 * @param directed Whether the graph is directed. If false, each edge is added in both directions.
 * @param default_weight The default weight to assign to edges if no weight is specified in the file.
 * @return The constructed Graph object.
 */
Graph read_graph_auto(const std::string& filename, bool directed, Graph::weight_t default_weight) {
    // Check extensions for the simplest detection first
    if (ends_with(filename, ".gr") || ends_with(filename, ".gr,gz")) {
        return read_from_gr(filename, directed);
    }

    if (ends_with(filename, ".txt") || ends_with(filename, ".txt.gz")) {
        return read_from_txt(filename, directed, default_weight);
    }

    // Unknown extension, attempt to auto-detect.
    std::cerr << "Warning: Unknown file extension for " << filename << ". Attempting to auto-detect format.\n";

    // Failback: Check the first non-comment line
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;

        char c = line[0];
        if (c == 'c' || c == '#' || c == '%') {
            // comment line
            continue;
        }
        if (c == 'p' || c == 'a') {
            // DIMACS format
            return read_from_gr(filename, directed);
        }
        else {
            // Assume edge list format
            return read_from_txt(filename, directed, default_weight);
        }
    }

    return Graph(); // empty graph
}