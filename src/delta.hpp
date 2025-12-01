/**
 * @file delta.hpp
 * @brief Declaration of Delta-Stepping SSSP algorithm using CUDA.
 * @author Daniel Zhao
 * @date 2025-11-30
 */
#pragma once

#include "graph.hpp"

#ifdef __CUDACC__

/**
 * @brief Kernel to relax light edges in Delta-Stepping algorithm.
 * @param g The graph in device format.
 * @param dist The distance array.
 * @param frontier The current frontier of vertices to process.
 * @param frontier_size The size of the current frontier.
 * @param delta The delta parameter for Delta-Stepping.
 * @param next_frontier The next frontier to populate.
 * @param next_frontier_size The size of the next frontier.
 * @note This kernel processes only light edges (edges with weight <= delta).
 * @note This kernel may insert duplicate vertices into the next frontier.
 */
__global__
void delta_relax_light(const Graph::DeviceGraph g, float* __restrict__ dist, const int* __restrict__ frontier, int frontier_size, float delta, int* __restrict__ next_frontier, int* __restrict__ next_frontier_size);

/**
 * @brief Kernel to relax heavy edges in Delta-Stepping algorithm.
 * @param g The graph in device format.
 * @param dist The distance array.
 * @param R The current bucket of vertices to process.
 * @param R_size The size of the current bucket.
 * @param delta The delta parameter for Delta-Stepping.
 * @param next_frontier The next frontier to populate.
 * @param next_frontier_size The size of the next frontier.
 * @note This kernel processes only heavy edges (edges with weight > delta).
 */
__global__
void delta_relax_heavy(const Graph::DeviceGraph g, float* __restrict__ dist, const int* __restrict__ R, int R_size, float delta, int* __restrict__ next_frontier, int* __restrict__ next_frontier_size);

#endif // __CUDACC__

/**
 * @brief Host function to execute Delta-Stepping SSSP algorithm on the GPU.
 * @param graph The input graph in host format.
 * @param source The source vertex for SSSP.
 * @param delta The delta parameter for Delta-Stepping.
 * @param gpu_init_time_seconds Optional pointer to a double to store the GPU initialization time in seconds.
 * @param gpu_compute_time_seconds Optional pointer to a double to store the GPU computation time in seconds.
 * @return A vector of shortest distances from the source to each vertex.
 */
std::vector<Graph::weight_t> delta_stepping_sssp(const Graph& graph, Graph::index_t source, float delta, double* gpu_init_time_seconds = nullptr, double* gpu_compute_time_seconds = nullptr);