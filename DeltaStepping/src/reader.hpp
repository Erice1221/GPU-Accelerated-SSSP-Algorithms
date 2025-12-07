/**
 * @file reader.hpp
 * @brief Declaration of graph reading utilities for GPU-accelerated SSSP algorithms.
 * @author Daniel Zhao
 * @date 2025-11-30
 */
#pragma once

#include <string>

#include "graph.hpp"

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
Graph read_from_txt(const std::string& filename, bool directed = true, Graph::weight_t default_weight = 1.0);

/**
 * @brief Reads a graph from a .gr file in DIMACS format. 
 * The file should contain comments starting with a `c`. a problem line starting with `p` that includes the number of vertices and edges,
 * and the edges starting with `a` in the format: source destination weight.
 * source and destinations are 1-indexed in the file; the function will reindex them to be zero-indexed.
 * @param filename The path to the .gr file containing the graph.
 * @param directed Whether the graph is directed. If false, each edge is added in both directions.
 * @return The constructed Graph object.
 */
Graph read_from_gr(const std::string& filename, bool directed = true);

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
Graph read_graph_auto(const std::string& filename, bool directed = true, Graph::weight_t default_weight = 1.0);