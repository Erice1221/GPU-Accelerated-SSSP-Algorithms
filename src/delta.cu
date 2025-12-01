/**
 * @file delta.cu
 * @brief Implementation of Delta-Stepping SSSP algorithm using CUDA.
 * @author Daniel Zhao
 * @date 2025-11-30
 */

#include <chrono>

#include "delta.hpp"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>


__device__ __constant__ float DELTA_STEPPING_INFINITY = 1e30f;

static __device__ inline float atomicMinFloat(float* address, float val) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        if (old_f <= val) break;
        int new_i = __float_as_int(val);
        old = atomicCAS(address_as_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
}

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
void delta_relax_light(const Graph::DeviceGraph g, float* __restrict__ dist, const int* __restrict__ frontier, int frontier_size, float delta, int* __restrict__ next_frontier, int* __restrict__ next_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    if (u < 0 || u >= g.num_vertices) return;

    float du = dist[u];
    if (du >= DELTA_STEPPING_INFINITY) return;

    int row_start = g.d_row_offsets[u];
    int row_end = g.d_row_offsets[u + 1];

    for (int e = row_start; e < row_end; ++e) {
        int v = g.d_col_indices[e];
        float w = g.d_edge_weights[e];

        if (w > delta) continue; // Only process light edges

        float new_dist = du + w;
        float old_dist = atomicMinFloat(&dist[v], new_dist);

        if (new_dist < old_dist) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = v;
        }
    }
}

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
void delta_relax_heavy(const Graph::DeviceGraph g, float* __restrict__ dist, const int* __restrict__ R, int R_size, float delta, int* __restrict__ next_frontier, int* __restrict__ next_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= R_size) return;

    int u = R[tid];
    if (u < 0 || u >= g.num_vertices) return;

    float du = dist[u];
    if (du >= DELTA_STEPPING_INFINITY) return;

    int row_start = g.d_row_offsets[u];
    int row_end = g.d_row_offsets[u + 1];

    for (int e = row_start; e < row_end; ++e) {
        int v = g.d_col_indices[e];
        float w = g.d_edge_weights[e];

        if (w <= delta) continue; // Only process heavy edges

        float new_dist = du + w;
        float old_dist = atomicMinFloat(&dist[v], new_dist);

        if (new_dist < old_dist) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = v;
        }
    }
}

/**
 * @brief Helper function to check for CUDA errors.
 * @param err The CUDA error code.
 * @param msg The message to display on error.
 */
static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::string m = std::string(msg) + ": " + cudaGetErrorString(err);
        throw std::runtime_error(m);
    }
}

/**
 * @brief Host function to execute Delta-Stepping SSSP algorithm on the GPU.
 * @param graph The input graph in host format.
 * @param source The source vertex for SSSP.
 * @param delta The delta parameter for Delta-Stepping.
 * @param gpu_init_time_seconds Optional pointer to a double to store the GPU initialization time in seconds.
 * @param gpu_compute_time_seconds Optional pointer to a double to store the GPU computation time in seconds.
 * @return A vector of shortest distances from the source to each vertex.
 */
std::vector<Graph::weight_t> delta_stepping_sssp(const Graph& graph, Graph::index_t source, float delta, double* gpu_init_time_seconds, double* gpu_compute_time_seconds) {
    using index_t = Graph::index_t;
    using weight_t = Graph::weight_t;

    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;

    auto total_start_time = clock::now();

    const int n = graph.num_vertices();
    if (n <= 0) return {};
    if (source < 0 || source >= n) throw std::runtime_error("delta_stepping_sssp: source vertex out of bounds");

    const weight_t INF = std::numeric_limits<weight_t>::infinity();

    // Host distance array
    std::vector<weight_t> h_dist(static_cast<std::size_t>(n), INF);
    h_dist[static_cast<std::size_t>(source)] = 0.0f;

    // Copy graph to device
    Graph::DeviceGraph d_graph = graph.copy_to_device();

    // Device distance array
    weight_t* d_dist = nullptr;
    check_cuda(cudaMalloc(&d_dist, n * sizeof(weight_t)), "cudaMalloc d_dist");
    check_cuda(cudaMemcpy(d_dist, h_dist.data(), n * sizeof(weight_t), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_dist");

    // Device frontier buffers
    int* d_frontier = nullptr;
    int* d_next_frontier = nullptr;
    int* d_next_frontier_size = nullptr;
    check_cuda(cudaMalloc(&d_frontier, n * sizeof(int)), "cudaMalloc d_frontier");
    check_cuda(cudaMalloc(&d_next_frontier, n * sizeof(int)), "cudaMalloc d_next_frontier");
    check_cuda(cudaMalloc(&d_next_frontier_size, sizeof(int)), "cudaMalloc d_next_frontier_size");

    // Host bucket structure
    std::vector<std::vector<int>> buckets;
    auto bucket_of = [delta](float d) -> int {
        if (d < 0.0f || !std::isfinite(d)) return -1;
        return static_cast<int>(d / delta);
    };

    // Initialize buckets
    {
        int b = bucket_of(h_dist[static_cast<std::size_t>(source)]);
        if (b < 0) b = 0;
        if (static_cast<std::size_t>(b) >= buckets.size()) buckets.resize(static_cast<std::size_t>(b) + 1);
        buckets[static_cast<std::size_t>(b)].push_back(static_cast<int>(source));
    }

    int current_bucket = 0;
    int max_bucket = static_cast<int>(buckets.size()) - 1;

    const int BLOCK_SIZE = 256;
    
    auto compute_start_time = clock::now();

    // Main Delta-Stepping loop
    while (current_bucket <= max_bucket) {
        if (current_bucket < 0 || static_cast<std::size_t>(current_bucket) >= buckets.size() || buckets[static_cast<std::size_t>(current_bucket)].empty()) {
            current_bucket++;
            continue;
        }

        std::vector<int> R;

        // Light edge relaxation phase
        while (!buckets[static_cast<std::size_t>(current_bucket)].empty()) {
            // Check current bucket
            std::vector<int> frontier;
            frontier.swap(buckets[static_cast<std::size_t>(current_bucket)]);
            if (frontier.empty()) break;
            
            // Insert to R
            R.insert(R.end(), frontier.begin(), frontier.end());

            int frontier_size = static_cast<int>(frontier.size());

            // Copy frontier to device
            check_cuda(cudaMemcpy(d_frontier, frontier.data(), frontier_size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D d_frontier");

            // Reset next frontier size
            int zero = 0;
            check_cuda(cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy reset d_next_frontier_size");

            // Launch light edge relaxation kernel
            int grid_size = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            delta_relax_light<<<grid_size, BLOCK_SIZE>>>(d_graph, d_dist, d_frontier, frontier_size, delta, d_next_frontier, d_next_frontier_size);
            check_cuda(cudaDeviceSynchronize(), "delta_relax_light sync");

            // Read back next frontier size
            int h_next_size = 0;
            check_cuda(cudaMemcpy(&h_next_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy next_frontier_size D2H");
            if (h_next_size <= 0) continue;

            // Read back distances and next frontier
            check_cuda(cudaMemcpy(h_dist.data(), d_dist, n * sizeof(weight_t), cudaMemcpyDeviceToHost), "cudaMemcpy dist D2H");
            std::vector<int> h_next_frontier(static_cast<std::size_t>(h_next_size));
            check_cuda(cudaMemcpy(h_next_frontier.data(), d_next_frontier, h_next_size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy next_frontier D2H");

            // Place vertices into appropriate buckets
            for (int v : h_next_frontier) {
                if (v < 0 || v >= n) continue;
                weight_t dv = h_dist[static_cast<std::size_t>(v)];
                if (!std::isfinite(dv) || dv == INF) continue;

                int b = bucket_of(dv);
                if (b < 0) continue;
                if (b < current_bucket) b = current_bucket;

                if (static_cast<std::size_t>(b) >= buckets.size()) buckets.resize(static_cast<std::size_t>(b) + 1);
                buckets[static_cast<std::size_t>(b)].push_back(v);
                if (b > max_bucket) max_bucket = b;
            }
        }

        // Heavy edge relaxation phase
        if (!R.empty()) {
            int R_size = static_cast<int>(R.size());

            // Copy R to device
            check_cuda(cudaMemcpy(d_frontier, R.data(), R_size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D R");

            // Reset next frontier size
            int zero = 0;
            check_cuda(cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy reset d_next_frontier_size (heavy)");

            // Launch heavy edge relaxation kernel
            int grid_size = (R_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            delta_relax_heavy<<<grid_size, BLOCK_SIZE>>>(d_graph, d_dist, d_frontier, R_size, delta, d_next_frontier, d_next_frontier_size);
            check_cuda(cudaDeviceSynchronize(), "delta_relax_heavy sync");

            // Read back next frontier size
            int h_next_size = 0;
            check_cuda(cudaMemcpy(&h_next_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy next_frontier_size D2H (heavy)");

            if (h_next_size > 0) {
                // Copy back distances and next frontier
                check_cuda(cudaMemcpy(h_dist.data(), d_dist, n * sizeof(weight_t), cudaMemcpyDeviceToHost), "cudaMemcpy dist D2H (heavy)");
                std::vector<int> h_next_frontier(static_cast<std::size_t>(h_next_size));
                check_cuda(cudaMemcpy(h_next_frontier.data(), d_next_frontier, h_next_size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy next_frontier D2H (heavy)");

                // Place vertices into appropriate buckets
                for (int v : h_next_frontier) {
                    if (v < 0 || v >= n) continue;
                    weight_t dv = h_dist[static_cast<std::size_t>(v)];
                    if (!std::isfinite(dv) || dv == INF) continue;

                    int b = bucket_of(dv);
                    if (b < 0) continue;
                    if (b <= current_bucket) b = current_bucket + 1;

                    if (static_cast<std::size_t>(b) >= buckets.size()) buckets.resize(static_cast<std::size_t>(b) + 1);
                    buckets[static_cast<std::size_t>(b)].push_back(v);
                    if (b > max_bucket) max_bucket = b;
                }
            }
        }
        ++current_bucket;
    }

    auto compute_end_time = clock::now();

    // Copy final distances back to host
    check_cuda(cudaMemcpy(h_dist.data(), d_dist, n * sizeof(weight_t), cudaMemcpyDeviceToHost), "cudaMemcpy final dist D2H");

    auto total_end_time = clock::now();
    if (gpu_compute_time_seconds) {
        *gpu_compute_time_seconds = seconds(compute_end_time - compute_start_time).count();
    }
    if (gpu_init_time_seconds) {
        double pre = seconds(compute_start_time - total_start_time).count();
        double post = seconds(total_end_time - compute_end_time).count();
        *gpu_init_time_seconds = pre + post;
    }

    // Cleanup
    Graph::free_device_graph(d_graph);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);

    return h_dist;
}

#endif // __CUDACC__