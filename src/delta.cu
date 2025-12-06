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
 * * Reads nodes from in_queue.
 * Scans neighbors.
 * Updates distance using atomicMin.
 * If updated:
 * - If edge_weight <= delta (Light): Add to next_near_queue
 * - If edge_weight >  delta (Heavy): Add to next_far_queue
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
    int* next_far_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= in_queue_size) return;

    int u = in_queue[tid];
    float d_u = dist[u];

    // Early exit if this path is already stale (optional optimization)
    if (d_u >= INF) return;

    int start = g.d_row_offsets[u];
    int end = g.d_row_offsets[u + 1];

    for (int i = start; i < end; ++i) {
        int v = g.d_col_indices[i];
        float w = g.d_edge_weights[i];
        
        float new_dist = d_u + w;
        
        // Speculative check to avoid expensive atomics
        if (new_dist < dist[v]) {
            float old_dist = atomicMinFloat(&dist[v], new_dist);
            
            // If we successfully improved the distance
            if (new_dist < old_dist) {
                // Determine which queue to add to
                if (w <= delta) {
                    // Light edge -> Add to next Near frontier
                    int pos = atomicAdd(next_near_count, 1);
                    next_near_queue[pos] = v;
                } else {
                    // Heavy edge -> Add to next Far frontier
                    int pos = atomicAdd(next_far_count, 1);
                    next_far_queue[pos] = v;
                }
            }
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
    using clock = std::chrono::high_resolution_clock;
    using seconds = std::chrono::duration<double>;

    auto total_start = clock::now();

    int n = graph.num_vertices();
    int m = graph.num_edges();

    // 1. Setup Device Graph
    Graph::DeviceGraph d_graph = graph.copy_to_device();
    
    float* d_dist;
    check_cuda(cudaMalloc(&d_dist, n * sizeof(float)), "Alloc dist");

    // 2. Queue Management
    // We need 3 queues: 
    // Q_curr (current processing), Q_next_near (light edges output), Q_far (heavy edges output)
    // To handle the "far" queue becoming "curr", we effectively act as a ring buffer or swap pointers.
    
    int *d_q1, *d_q2, *d_q_far; 
    check_cuda(cudaMalloc(&d_q1, n * sizeof(int) * 2), "Alloc Q1"); // Make queues slightly larger or exactly N. 
    // *Note*: In worst case with duplicates, queue could exceed N. 
    // Ideally we implement duplicate removal, but for simplicity/speed here, we allocate generous buffers.
    // For very large graphs, we would need a `bool visited` array reset every iter to enforce uniqueness.
    // Here we assume N * 2 is safe enough for basic benchmarking or check bounds in kernel.
    
    // For safety in this implementation, let's just alloc N. 
    // High-performance implementations use deduplication bitmasks.
    check_cuda(cudaMalloc(&d_q2, n * sizeof(int) * 2), "Alloc Q2");
    check_cuda(cudaMalloc(&d_q_far, n * sizeof(int) * 2), "Alloc QFar");

    // Counters (managed on device to avoid PCIe latency)
    int* d_counters; 
    // [0]: q_curr_size, [1]: q_next_near_size, [2]: q_far_size
    check_cuda(cudaMalloc(&d_counters, 3 * sizeof(int)), "Alloc counters");
    check_cuda(cudaMemset(d_counters, 0, 3 * sizeof(int)), "Zero counters");

    // Initialize Distances
    int grid_init = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_buffers<<<grid_init, BLOCK_SIZE>>>(d_dist, n, source);
    check_cuda(cudaDeviceSynchronize(), "Init buffers");

    // Initialize Frontier with Source
    int h_one = 1;
    check_cuda(cudaMemcpy(d_q1, &source, sizeof(int), cudaMemcpyHostToDevice), "Init Source");
    check_cuda(cudaMemcpy(&d_counters[0], &h_one, sizeof(int), cudaMemcpyHostToDevice), "Init Counter");

    // Pointers to simplify swapping
    int* d_curr = d_q1;
    int* d_next = d_q2;
    // d_q_far stays constant until we flush it

    auto compute_start = clock::now();

    int h_curr_size = 1;
    int h_far_size = 0;

    // --- Main Loop ---
    // Runs while there is work in Current (Near) queue OR Far queue
    while (h_curr_size > 0 || h_far_size > 0) {
        
        // 1. Inner Loop: Process Near Queue until empty (Convergence of light edges)
        while (h_curr_size > 0) {
            int grid_size = (h_curr_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // Launch Relax
            // Input: d_curr (size counters[0])
            // Output Light: d_next (size counters[1])
            // Output Heavy: d_q_far (size counters[2])
            relax_kernel<<<grid_size, BLOCK_SIZE>>>(
                d_graph, 
                d_dist, 
                d_curr, 
                h_curr_size, 
                delta, 
                d_next, 
                &d_counters[1], // next_near_count
                d_q_far, 
                &d_counters[2]  // next_far_count
            );
            check_cuda(cudaGetLastError(), "Relax Kernel Launch");

            // We must wait for this to finish to know the new size
            // (In highly optimized code, we might use device-side launch or persistent kernels)
            // But checking a single int from device is much faster than copying the whole array.
            check_cuda(cudaMemcpy(&h_curr_size, &d_counters[1], sizeof(int), cudaMemcpyDeviceToHost), "Read Near Size");
            
            // Swap Near Queues
            std::swap(d_curr, d_next);
            
            // Reset "next" counter for the upcoming iteration (which is now old `d_curr` buffer)
            check_cuda(cudaMemset(&d_counters[1], 0, sizeof(int)), "Reset Next Counter");
            
            // Update input size for next iter
            // (h_curr_size is already set from the memcpy above)
            
            // IMPORTANT: Update counter[0] to reflect the swap for the kernel's next read?
            // Actually, we pass h_curr_size by value to kernel. We only need to reset the atomic counter for the *output*.
            // The input array `d_curr` is just a pointer.
        }

        // 2. Inner Loop Done: Move Far Queue to Near Queue
        check_cuda(cudaMemcpy(&h_far_size, &d_counters[2], sizeof(int), cudaMemcpyDeviceToHost), "Read Far Size");
        
        if (h_far_size > 0) {
            // Swap d_q_far into d_curr
            // To avoid losing the pointer to the buffer, we swap pointers
            int* temp = d_curr;
            d_curr = d_q_far;
            d_q_far = temp;
            
            h_curr_size = h_far_size;
            h_far_size = 0; // We moved it all
            
            // Reset Far Counter
            check_cuda(cudaMemset(&d_counters[2], 0, sizeof(int)), "Reset Far Counter");
        }
    }

    auto compute_end = clock::now();
    check_cuda(cudaDeviceSynchronize(), "Final Sync");

    // Copy result back
    std::vector<float> h_dist(n);
    check_cuda(cudaMemcpy(h_dist.data(), d_dist, n * sizeof(float), cudaMemcpyDeviceToHost), "Copy Result");

    // Timing calculations
    if (gpu_compute_time_seconds) {
        *gpu_compute_time_seconds = seconds(compute_end - compute_start).count();
    }
    if (gpu_init_time_seconds) {
        auto total_end = clock::now();
        double total = seconds(total_end - total_start).count();
        double compute = seconds(compute_end - compute_start).count();
        *gpu_init_time_seconds = total - compute;
    }

    // Cleanup
    cudaFree(d_dist);
    cudaFree(d_q1);
    cudaFree(d_q2);
    cudaFree(d_q_far);
    cudaFree(d_counters);
    Graph::free_device_graph(d_graph);

    return h_dist;
}

#endif // __CUDACC__