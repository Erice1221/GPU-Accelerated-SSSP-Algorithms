/**
 * @file dijkstra.cpp
 * @brief Implementation of Dijkstra's algorithm on the CPU. Used as reference baseline for GPU implementations.
 * @author Daniel Zhao
 * @date 2025-11-30
 */

#include <queue>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <chrono>

#include "dijkstra.hpp"

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
std::vector<Graph::weight_t> dijkstra_cpu(const Graph& graph, Graph::index_t source, std::vector<Graph::index_t>* parent, double* cpu_time_seconds) {
    using index_t = Graph::index_t;
    using weight_t = Graph::weight_t;

    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;

    auto start_time = clock::now();

    const index_t n = graph.num_vertices();

    if (source < 0 || source >= n) {
        throw std::runtime_error("dijkstra_cpu: source vertex out of range");
    }

    const weight_t INF = std::numeric_limits<weight_t>::infinity();
    std::vector<weight_t> dist(static_cast<std::size_t>(n), INF);

    // Optional parent tracking
    if (parent) {
        parent->assign(static_cast<std::size_t>(n), static_cast<index_t>(-1));
    }

    dist[static_cast<std::size_t>(source)] = static_cast<weight_t>(0);

    // Priority queue for selecting the next vertex with the smallest distance
    struct HeapNode {
        weight_t dist;
        index_t  v;
    };

    struct Compare {
        bool operator()(const HeapNode& a, const HeapNode& b) const {
            // priority_queue is max-heap by default; reverse for min-heap
            return a.dist > b.dist;
        }
    };

    std::priority_queue<HeapNode, std::vector<HeapNode>, Compare> pq;
    pq.push({static_cast<weight_t>(0), source});

    const auto& row_offsets  = graph.row_offsets();
    const auto& col_indices  = graph.col_indices();
    const auto& edge_weights = graph.edge_weights();

    while (!pq.empty()) {
        HeapNode top = pq.top();
        pq.pop();

        const index_t u      = top.v;
        const weight_t d_u   = top.dist;
        const std::size_t u_idx = static_cast<std::size_t>(u);

        // Skip stale entries
        if (d_u > dist[u_idx]) {
            continue;
        }

        // Explore outgoing edges of u
        const index_t row_start = row_offsets[static_cast<std::size_t>(u)];
        const index_t row_end   = row_offsets[static_cast<std::size_t>(u) + 1];

        for (index_t e = row_start; e < row_end; ++e) {
            const std::size_t e_idx = static_cast<std::size_t>(e);
            const index_t v         = col_indices[e_idx];
            const weight_t w        = edge_weights[e_idx];

            // Dijkstra assumes non-negative weights; you may want to assert here.
            assert(w >= static_cast<weight_t>(0));
            const std::size_t v_idx = static_cast<std::size_t>(v);
            const weight_t new_dist = d_u + w;

            if (new_dist < dist[v_idx]) {
                dist[v_idx] = new_dist;
                if (parent) {
                    (*parent)[v_idx] = u;
                }
                pq.push({new_dist, v});
            }
        }
    }

    auto end_time = clock::now();
    if (cpu_time_seconds) {
        *cpu_time_seconds = seconds(end_time - start_time).count();
    }

    return dist;
}