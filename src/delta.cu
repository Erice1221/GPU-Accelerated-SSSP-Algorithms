/**
 * @file delta.cu
 * @brief Implementation of Delta-Stepping SSSP algorithm using CUDA.
 * @author Daniel Zhao
 * @date 2025-11-30
 */

#include <chrono>
#include <vector>
#include <iostream>
#include <limits>

#include "delta.hpp"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

#define INF 1e30f
#define BLOCK_SIZE 256

/**
 * @brief Helper function to check for CUDA errors.
 * @param err The CUDA error code.
 * @param msg The message to display on error.
 */
#define check_cuda(ans, msg) { gpuAssert((ans), __FILE__, __LINE__, msg); }
inline void gpuAssert(cudaError_t code, const char *file, int line, const char *msg) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " " 
                  << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        exit(1);
    }
}

/**
 * @brief Atomic minimum operation for floats.
 * @param address The address of the float to update.
 * @param val The value to compare and set.
 * @return The old value at the address.
 */
static __device__ inline float atomicMinFloat(float* address, float val) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        // If existing value is already smaller/equal, no need to update
        if (old_f <= val) return old_f;
        
        int new_i = __float_as_int(val);
        old = atomicCAS(address_as_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Initialization Kernel.
 */
__global__ void init_buffers(float* dist, int n, int source) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (tid == source) dist[tid] = 0.0f;
        else dist[tid] = INF;
    }
}

/**
 * @brief Relax Kernel.
 * @param g The device graph.
 * @param dist The distance array.
 * @param in_queue The current queue of nodes to process.
 * @param in_queue_size The size of the current queue.
 * @param delta The delta parameter.
 * @param next_near_queue The next near queue to populate.
 * @param next_near_count Pointer to the near queue size counter.
 * @param next_far_queue The next far queue to populate.
 * @param next_far_count Pointer to the far queue size counter.
 * @param in_next_near_map Deduplication array for near queue.
 * @param in_next_far_map Deduplication array for far queue.
 */
__global__ void relax_kernel(
    const Graph::DeviceGraph g, 
    float* dist, 
    const int* in_queue, 
    int in_queue_size,
    float delta,
    int* next_near_queue, 
    int* next_near_count,
    int* next_far_queue,
    int* next_far_count,
    int* in_next_near_map, // Bitmask/Map for near queue
    int* in_next_far_map   // Bitmask/Map for far queue
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < in_queue_size) {
        int u = in_queue[tid];
        float d_u = dist[u];

        // Sanity check
        if (d_u < INF) {
            int start = g.d_row_offsets[u];
            int end = g.d_row_offsets[u + 1];

            for (int i = start; i < end; ++i) {
                int v = g.d_col_indices[i];
                float w = g.d_edge_weights[i];
                float new_dist = d_u + w;

                if (new_dist < dist[v]) {
                    // Attempt to update distance
                    float old_dist = atomicMinFloat(&dist[v], new_dist);
                    
                    // If WE were the ones to lower the distance (or it was lowered)
                    if (new_dist < old_dist) {
                        
                        // Decide which bucket (Near vs Far)
                        if (w <= delta) {
                            // DEDUPLICATION:
                            // Try to set the flag for 'v' in the map.
                            // atomicExch returns the OLD value.
                            // If old value was 0, it means we are the FIRST thread to add 'v' this round.
                            // If old value was 1, 'v' is already scheduled to be added, so we skip adding.
                            if (atomicExch(&in_next_near_map[v], 1) == 0) {
                                int pos = atomicAdd(next_near_count, 1);
                                next_near_queue[pos] = v;
                            }
                        } else {
                            if (atomicExch(&in_next_far_map[v], 1) == 0) {
                                int pos = atomicAdd(next_far_count, 1);
                                next_far_queue[pos] = v;
                            }
                        }
                    }
                }
            }
        }
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
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;

    auto total_start = clock::now();

    // 1. Setup Device Graph
    Graph::DeviceGraph d_graph = graph.copy_to_device();
    int n = graph.num_vertices();

    // 2. Allocate Memory
    float* d_dist;
    check_cuda(cudaMalloc(&d_dist, n * sizeof(float)), "Malloc Dist");

    int *d_curr, *d_next_near, *d_next_far;
    check_cuda(cudaMalloc(&d_curr, n * sizeof(int)), "Malloc Curr");
    check_cuda(cudaMalloc(&d_next_near, n * sizeof(int)), "Malloc Near");
    check_cuda(cudaMalloc(&d_next_far, n * sizeof(int)), "Malloc Far");

    // Deduplication Maps (Init to 0)
    int *d_in_near_map, *d_in_far_map;
    check_cuda(cudaMalloc(&d_in_near_map, n * sizeof(int)), "Malloc Near Map");
    check_cuda(cudaMalloc(&d_in_far_map, n * sizeof(int)), "Malloc Far Map");
    check_cuda(cudaMemset(d_in_near_map, 0, n * sizeof(int)), "Init Near Map");
    check_cuda(cudaMemset(d_in_far_map, 0, n * sizeof(int)), "Init Far Map");

    // Counters: [0]=curr_size, [1]=near_size, [2]=far_size
    int* d_counters; 
    check_cuda(cudaMalloc(&d_counters, 3 * sizeof(int)), "Malloc Counters");
    check_cuda(cudaMemset(d_counters, 0, 3 * sizeof(int)), "Zero Counters");

    // 3. Initialization
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_buffers<<<blocks, BLOCK_SIZE>>>(d_dist, n, source);
    check_cuda(cudaDeviceSynchronize(), "Init Buffers");

    // Add source to current queue
    check_cuda(cudaMemcpy(d_curr, &source, sizeof(int), cudaMemcpyHostToDevice), "Copy Source");
    int h_curr_size = 1;
    check_cuda(cudaMemcpy(&d_counters[0], &h_curr_size, sizeof(int), cudaMemcpyHostToDevice), "Set Source Size");

    auto compute_start = clock::now();

    int h_near_size = 0;
    int h_far_size = 0;

    // 4. Main Loop
    while (h_curr_size > 0 || h_far_size > 0) {
        
        // --- Inner Loop: Light Edges (Delta) ---
        while (h_curr_size > 0) {
            int grid_size = (h_curr_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            relax_kernel<<<grid_size, BLOCK_SIZE>>>(
                d_graph, d_dist, 
                d_curr, h_curr_size, delta, 
                d_next_near, &d_counters[1], 
                d_next_far, &d_counters[2],
                d_in_near_map, d_in_far_map
            );
            check_cuda(cudaGetLastError(), "Relax Kernel");

            // Read counters
            int h_counters[3];
            check_cuda(cudaMemcpy(h_counters, d_counters, 3 * sizeof(int), cudaMemcpyDeviceToHost), "Read Counters");
            h_near_size = h_counters[1];
            h_far_size = h_counters[2];

            // Move Next Near -> Curr
            int* temp = d_curr;
            d_curr = d_next_near;
            d_next_near = temp;

            h_curr_size = h_near_size;

            // Reset Near Counter
            check_cuda(cudaMemset(&d_counters[1], 0, sizeof(int)), "Reset Near Counter");
            
            // CRITICAL: We just processed 'Near', so the 'Near Map' is now stale (those nodes are in Curr).
            // We clear it so they can be added again if needed.
            check_cuda(cudaMemset(d_in_near_map, 0, n * sizeof(int)), "Reset Near Map");
        }

        // --- Outer Loop: Heavy Edges ---
        // Refetch Far Size
        check_cuda(cudaMemcpy(&h_far_size, &d_counters[2], sizeof(int), cudaMemcpyDeviceToHost), "Read Far Size");
        
        if (h_far_size > 0) {
            // Move Far -> Curr
            int* temp = d_curr;
            d_curr = d_next_far;
            d_next_far = temp;
            
            h_curr_size = h_far_size;
            h_far_size = 0;
            
            // Reset Far Counter & Map
            check_cuda(cudaMemset(&d_counters[2], 0, sizeof(int)), "Reset Far Counter");
            check_cuda(cudaMemset(d_in_far_map, 0, n * sizeof(int)), "Reset Far Map");
        }
    }

    auto compute_end = clock::now();
    check_cuda(cudaDeviceSynchronize(), "Final Sync");

    // Copy result back
    std::vector<float> h_dist(n);
    check_cuda(cudaMemcpy(h_dist.data(), d_dist, n * sizeof(float), cudaMemcpyDeviceToHost), "Copy Result");

    // Cleanup
    cudaFree(d_dist);
    cudaFree(d_curr);
    cudaFree(d_next_near);
    cudaFree(d_next_far);
    cudaFree(d_counters);
    cudaFree(d_in_near_map);
    cudaFree(d_in_far_map);
    Graph::free_device_graph(d_graph);

    if (gpu_compute_time_seconds) {
        *gpu_compute_time_seconds = seconds(compute_end - compute_start).count();
    }
    if (gpu_init_time_seconds) {
        auto total_end = clock::now();
        double total = seconds(total_end - total_start).count();
        double compute = seconds(compute_end - compute_start).count();
        *gpu_init_time_seconds = total - compute;
    }

    return h_dist;
}

#endif // __CUDACC__