/**
 * @file graph.hpp
 * @brief Declaration of Graph structure for GPU-accelerated SSSP algorithms.
 * @author Daniel Zhao
 * @date 2025-11-30
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <utility>
#include <stdexcept>

/**
 * @brief Represents a graph structure for GPU-accelerated SSSP algorithms. 
 * We would use a compressed sparse row (CSR) format for efficient storage and access.
 */
class Graph {
    public:
        using index_t = std::int32_t;
        using weight_t = float;
        
        Graph() : num_vertices_(0), num_edges_(0) {}

        Graph(index_t num_vertices, const std::vector<std::pair<index_t, index_t>>& edges, const std::vector<weight_t>& weights) {
            build_from_edge_list(num_vertices, edges, weights);
        }

        void build_from_edge_list(index_t num_vertices, const std::vector<std::pair<index_t, index_t>>& edges, const std::vector<weight_t>& weights) {
            if (edges.size() != weights.size()) {
                throw std::runtime_error("edges.size() != weights.size()");
            }

            num_vertices_ = num_vertices;
            num_edges_    = static_cast<index_t>(edges.size());

            row_offsets_.assign(static_cast<std::size_t>(num_vertices_) + 1, 0);
            col_indices_.resize(static_cast<std::size_t>(num_edges_));
            edge_weights_.resize(static_cast<std::size_t>(num_edges_));

            // Count outgoing degree for each vertex
            for (const auto& e : edges) {
                index_t u = e.first;
                if (u < 0 || u >= num_vertices_) {
                    throw std::runtime_error("Edge source out of range");
                }
                row_offsets_[static_cast<std::size_t>(u) + 1]++;
            }

            // Prefix sum to get row_offsets
            for (index_t v = 0; v < num_vertices_; ++v) {
                row_offsets_[static_cast<std::size_t>(v) + 1] += row_offsets_[static_cast<std::size_t>(v)];
            }

            // Temporary copy to track "insertion positions"
            std::vector<index_t> next_offset(row_offsets_.begin(), row_offsets_.end());

            // Fill CSR arrays
            for (std::size_t i = 0; i < edges.size(); ++i) {
                index_t u = edges[i].first;
                index_t v = edges[i].second;

                if (v < 0 || v >= num_vertices_) {
                    throw std::runtime_error("Edge destination out of range");
                }

                index_t pos = next_offset[static_cast<std::size_t>(u)]++;
                col_indices_[static_cast<std::size_t>(pos)] = v;
                edge_weights_[static_cast<std::size_t>(pos)] = weights[i];
            }
        }

        index_t num_vertices() const noexcept { return num_vertices_; }
        index_t num_edges() const noexcept { return num_edges_; }

        const std::vector<index_t>& row_offsets() const noexcept { return row_offsets_; }
        const std::vector<index_t>& col_indices() const noexcept { return col_indices_; }
        const std::vector<weight_t>& edge_weights() const noexcept { return edge_weights_; }

        std::pair<const index_t*, const index_t*> neighbors(index_t u) const {
            if (u < 0 || u >= num_vertices_) {
                throw std::runtime_error("neighbors: vertex out of range");
            }
            std::size_t start = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(u)]);
            std::size_t end   = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(u) + 1]);
            return { &col_indices_[start], &col_indices_[end] };
        }

    #ifdef __CUDACC__
        struct DeviceGraph {
            index_t num_vertices;
            index_t num_edges;
            index_t* d_row_offsets;
            int2* d_edges_interleaved; // .x = neighbor, .y = float_as_int(weight)
        };

        DeviceGraph copy_to_device() const {
            DeviceGraph d_graph;
            d_graph.num_vertices = num_vertices_;
            d_graph.num_edges    = num_edges_;

            std::size_t n_rows = static_cast<std::size_t>(num_vertices_) + 1;
            std::size_t n_edges = static_cast<std::size_t>(num_edges_);

            cudaError_t err;

            err = cudaMalloc(reinterpret_cast<void**>(&d_graph.d_row_offsets),
                            n_rows * sizeof(index_t));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc d_row_offsets failed");
            }

            err = cudaMalloc(reinterpret_cast<void**>(&d_graph.d_edges_interleaved),
                            n_edges * sizeof(int2));
            if (err != cudaSuccess) {
                cudaFree(d_graph.d_row_offsets);
                throw std::runtime_error("cudaMalloc d_edges_interleaved failed");
            }

            err = cudaMemcpy(d_graph.d_row_offsets,
                            row_offsets_.data(),
                            n_rows * sizeof(index_t),
                            cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(d_graph.d_row_offsets);
                cudaFree(d_graph.d_edges_interleaved);
                throw std::runtime_error("cudaMemcpy row_offsets failed");
            }

            // Interleave data on host before copying
            std::vector<int2> h_edges_interleaved(n_edges);
            for (size_t i = 0; i < n_edges; ++i) {
                h_edges_interleaved[i].x = col_indices_[i];
                // Reinterpret cast float weight to int for storage
                float w = edge_weights_[i];
                h_edges_interleaved[i].y = *reinterpret_cast<int*>(&w);
            }

            err = cudaMemcpy(d_graph.d_edges_interleaved,
                            h_edges_interleaved.data(),
                            n_edges * sizeof(int2),
                            cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(d_graph.d_row_offsets);
                cudaFree(d_graph.d_edges_interleaved);
                throw std::runtime_error("cudaMemcpy d_edges_interleaved failed");
            }

            return d_graph;
        }

        static void free_device_graph(DeviceGraph& d_graph) {
            if (d_graph.d_row_offsets) {
                cudaFree(d_graph.d_row_offsets);
                d_graph.d_row_offsets = nullptr;
            }
            if (d_graph.d_edges_interleaved) {
                cudaFree(d_graph.d_edges_interleaved);
                d_graph.d_edges_interleaved = nullptr;
            }
            d_graph.num_vertices = 0;
            d_graph.num_edges    = 0;
        }
    #endif // __CUDACC__

    private:
        index_t num_vertices_;
        index_t num_edges_;
        std::vector<index_t>  row_offsets_;
        std::vector<index_t>  col_indices_;
        std::vector<weight_t> edge_weights_;
};