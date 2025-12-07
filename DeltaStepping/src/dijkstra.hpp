/**
 * @file dijkstra.hpp
 * @brief Declaration of Dijkstra's algorithm for Single Source Shortest Path (SSSP) on the CPU.
 * @author Daniel Zhao
 * @date 2025-11-30
 */
#pragma once

#include <vector>

#include "graph.hpp"

/**
 * @brief Implements Dijkstra's algorithm for Single Source Shortest Path (SSSP) on the CPU.
 * @param graph The input graph in CSR format.
 * @param source The source vertex from which to calculate shortest paths.
 * @param parent Optional pointer to a vector to store the parent of each vertex in the shortest path tree.
 * @param cpu_time_seconds Optional pointer to a double to store the CPU execution time in seconds.
 * @return A vector of shortest path distances from the source to each vertex.
 * @pre All edge weights in the graph must be non-negative.
 * @pre source must be a valid vertex index in the graph (i.e., 0 <= source < graph.num_vertices()).
 */
std::vector<Graph::weight_t> dijkstra_cpu(const Graph& graph, Graph::index_t source, std::vector<Graph::index_t>* parent = nullptr, double* cpu_time_seconds = nullptr);