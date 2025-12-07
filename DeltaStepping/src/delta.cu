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
#define WARP_SIZE 32

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
        if (old_f <= val) return old_f;
        int new_i = __float_as_int(val);
        old = atomicCAS(address_as_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Global barrier synchronization across all blocks.
 * @param goal_val The number of blocks to wait for.
 * @param barrier_array Pointer to a global memory array used for synchronization.
 */
__device__ void global_sync(volatile unsigned long long* barrier_val, unsigned long long* local_goal, int num_blocks) {
    __threadfence(); // Ensure all writes are visible
    __syncthreads(); // Finish local work
    
    if (threadIdx.x == 0) {
        // Increment global counter
        // atomicAdd on volatile is not standard in older CUDA, so we cast away volatile for the atomic
        // but we assume the atomic operation implies memory consistency.
        atomicAdd((unsigned long long*)barrier_val, 1);
        
        // Wait until everyone reaches the CURRENT goal
        // STRICT READ from volatile memory
        while (*barrier_val < *local_goal) {
            // Busy wait
        }
    }
    __syncthreads(); // Wait for leader to confirm global sync
    
    // Update goal for NEXT sync
    if (threadIdx.x == 0) {
        *local_goal += num_blocks;
    }
    __syncthreads();
}

/**
 * @brief Warp-Aggregated Queue Append using __activemask()
 * Safe for both divergent and convergent control flow.
 */
__device__ inline void append_to_queue_cooperative(int v, int* queue, int* counter) {
    unsigned int active_mask = __activemask();
    unsigned int vote_mask = __ballot_sync(active_mask, 1);
    
    int agg_count = __popc(vote_mask);
    int lane_id = threadIdx.x % 32;
    int local_rank = __popc(vote_mask & ((1 << lane_id) - 1));
    
    int leader_lane = __ffs(vote_mask) - 1;
    int global_base_offset = 0;
    
    if (lane_id == leader_lane) {
        global_base_offset = atomicAdd(counter, agg_count);
    }
    
    global_base_offset = __shfl_sync(active_mask, global_base_offset, leader_lane);
    queue[global_base_offset + local_rank] = v;
}

/**
 * @brief Process a single edge and add to appropriate queue if relaxed.
 * @param v The target vertex.
 * @param w The edge weight.
 * @param d_u The distance to the source vertex.
 * @param delta The delta parameter.
 * @param dist The distance array.
 * @param next_near_queue The next near queue.
 * @param next_near_count Pointer to the near queue size counter.
 * @param next_far_queue The next far queue.
 * @param next_far_count Pointer to the far queue size counter.
 * @param in_next_near_map Deduplication array for near queue.
 * @param in_next_far_map Deduplication array for far queue.
 */
__device__ inline void process_edge(
    int v, float w, float d_u, float delta,
    float* dist,
    int* next_near_queue, int* next_near_count,
    int* next_far_queue, int* next_far_count,
    int* in_next_near_map, int* in_next_far_map,
    int near_gen, int far_gen
) {
    float new_dist = d_u + w;
    if (new_dist < dist[v]) {
        float old_dist = atomicMinFloat(&dist[v], new_dist);
        if (new_dist < old_dist) {
            if (w <= delta) {
                // Deduplication (Bitmasking)
                if (atomicExch(&in_next_near_map[v], near_gen) != near_gen) {
                    append_to_queue_cooperative(v, next_near_queue, next_near_count);
                }
            } else {
                if (atomicExch(&in_next_far_map[v], far_gen) != far_gen) {
                    append_to_queue_cooperative(v, next_far_queue, next_far_count);
                }
            }
        }
    }
}

/**
 * @brief Relaxation Kernel for Delta-Stepping.
 * @param g The device graph.
 * @param dist The distance array.
 * @param curr_queue The current queue.
 * @param curr_size The size of the current queue.
 * @param delta The delta parameter.
 * @param next_near_queue The next near queue.
 * @param next_near_count Pointer to the near queue size counter.
 * @param next_far_queue The next far queue.
 * @param next_far_count Pointer to the far queue size counter.
 * @param in_next_near_map Deduplication array for near queue.
 * @param in_next_far_map Deduplication array for far queue.
 */
__global__ void sssp_persistent_kernel(
    const Graph::DeviceGraph g, 
    float* __restrict__ dist, 
    int* __restrict__ d_curr, 
    int* __restrict__ d_next_near, 
    int* __restrict__ d_next_far,
    volatile int* d_counters, // Volatile is crucial
    float delta,
    int* __restrict__ d_in_near_map, 
    int* __restrict__ d_in_far_map,
    volatile unsigned long long* barrier_val,
    int num_blocks
) {
    int* q_curr = d_curr;
    int* q_next_near = d_next_near;
    int* q_next_far = d_next_far;
    
    unsigned long long local_barrier_goal = num_blocks;

    while (true) {
        // --- 1. SYNC START ---
        global_sync(barrier_val, &local_barrier_goal, num_blocks);

        int curr_size = d_counters[0];
        int near_gen = d_counters[3];
        int far_gen = d_counters[4];

        // --- 2. TERMINATION CHECK ---
        // Only break if we are completely empty and have no pending swaps
        if (curr_size == 0 && d_counters[2] == 0 && d_counters[1] == 0) {
             break; 
        }

        // --- 3. RELAXATION PHASE ---
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;

        if (curr_size > 0) {
            for (int i = tid; i < curr_size; i += stride) {
                int u = __ldg(&q_curr[i]);
                float d_u = dist[u];

                if (d_u < INF) {
                    int start = __ldg(&g.d_row_offsets[u]);
                    int end = __ldg(&g.d_row_offsets[u + 1]);
                    
                    bool is_heavy = (end - start) >= WARP_SIZE;
                    
                    if (!is_heavy) {
                        for (int j = start; j < end; ++j) {
                            int2 edge = __ldg(&g.d_edges_interleaved[j]);
                            int v = edge.x;
                            float w = __int_as_float(edge.y);
                            process_edge(v, w, d_u, delta, dist, 
                                        q_next_near, (int*)&d_counters[1], 
                                        q_next_far, (int*)&d_counters[2], 
                                        d_in_near_map, d_in_far_map,
                                        near_gen, far_gen);
                        }
                    } else {
                        unsigned int active_mask = __activemask();
                        unsigned int heavy_mask = __ballot_sync(active_mask, is_heavy);
                        int num_active = __popc(active_mask);
                        int lane = threadIdx.x % 32;
                        int rank = __popc(active_mask & ((1U << lane) - 1));

                        while (heavy_mask) {
                            int leader = __ffs(heavy_mask) - 1;
                            int l_start = __shfl_sync(active_mask, start, leader);
                            int l_end   = __shfl_sync(active_mask, end, leader);
                            float l_du  = __shfl_sync(active_mask, d_u, leader);
                            
                            for (int j = l_start + rank; j < l_end; j += num_active) {
                                int2 edge = __ldg(&g.d_edges_interleaved[j]);
                                int v = edge.x;
                                float w = __int_as_float(edge.y);
                                process_edge(v, w, l_du, delta, dist, 
                                            q_next_near, (int*)&d_counters[1], 
                                            q_next_far, (int*)&d_counters[2], 
                                            d_in_near_map, d_in_far_map,
                                            near_gen, far_gen);
                            }
                            heavy_mask &= ~(1U << leader);
                        }
                    }
                }
            }
        }
        
        // --- 4. BARRIER (Wait for work) ---
        global_sync(barrier_val, &local_barrier_goal, num_blocks);

        // --- 5. UNIFIED QUEUE MANAGEMENT ---
        // Everyone reads the counters to decide the 'Mode'
        int near_cnt = d_counters[1];
        int far_cnt = d_counters[2];
        
        int mode = 0; // 0=None, 1=Near->Curr, 2=Far->Curr
        if (near_cnt > 0) mode = 1;
        else if (far_cnt > 0) mode = 2;

        // Step A: Map Reset (Parallel)
        if (mode == 1) {
             if (tid == 0 && blockIdx.x == 0) d_counters[3]++;
        } else if (mode == 2) {
             if (tid == 0 && blockIdx.x == 0) d_counters[4]++;
        } else {
             // If mode is 0 (both empty), ensure curr is marked empty for next loop top check
             if (tid == 0 && blockIdx.x == 0) d_counters[0] = 0;
        }
        
        // --- 6. UNCONDITIONAL SYNC ---
        // This prevents the deadlock. Everyone waits here regardless of mode.
        global_sync(barrier_val, &local_barrier_goal, num_blocks);

        // Step B: Pointer Swapping (Master logic)
        // Pointers are local registers, so every thread must swap its own view
        if (mode == 1) {
            int* temp = q_curr; q_curr = q_next_near; q_next_near = temp;
            if (tid == 0 && blockIdx.x == 0) {
                d_counters[0] = near_cnt;
                d_counters[1] = 0;
            }
        } else if (mode == 2) {
            int* temp = q_curr; q_curr = q_next_far; q_next_far = temp;
            if (tid == 0 && blockIdx.x == 0) {
                d_counters[0] = far_cnt;
                d_counters[2] = 0;
            }
        }
    }
}

/**
 * @brief Initialization Kernel.
 * @param dist The distance array.
 * @param n Number of vertices.
 * @param source The source vertex.
 */
__global__ void init_buffers(float* dist, int n, int source) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (tid == source) dist[tid] = 0.0f;
        else dist[tid] = INF;
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
    Graph::DeviceGraph d_graph = graph.copy_to_device();

    // Allocations
    float* d_dist;
    check_cuda(cudaMalloc(&d_dist, n * sizeof(float)), "Malloc Dist");

    int *d_curr, *d_next_near, *d_next_far;
    check_cuda(cudaMalloc(&d_curr, n * sizeof(int)), "Malloc Curr");
    check_cuda(cudaMalloc(&d_next_near, n * sizeof(int)), "Malloc Near");
    check_cuda(cudaMalloc(&d_next_far, n * sizeof(int)), "Malloc Far");

    int *d_in_near_map, *d_in_far_map;
    check_cuda(cudaMalloc(&d_in_near_map, n * sizeof(int)), "Malloc Near Map");
    check_cuda(cudaMalloc(&d_in_far_map, n * sizeof(int)), "Malloc Far Map");
    check_cuda(cudaMemset(d_in_near_map, 0, n * sizeof(int)), "Init Near Map");
    check_cuda(cudaMemset(d_in_far_map, 0, n * sizeof(int)), "Init Far Map");

    int* d_counters; 
    check_cuda(cudaMalloc(&d_counters, 8 * sizeof(int)), "Malloc Counters");
    check_cuda(cudaMemset(d_counters, 0, 8 * sizeof(int)), "Zero Counters");

    // Initialize generations to 1 so they don't match the initial 0 in maps
    int initial_gen = 1;
    check_cuda(cudaMemcpy(&d_counters[3], &initial_gen, sizeof(int), cudaMemcpyHostToDevice), "Init Near Gen");
    check_cuda(cudaMemcpy(&d_counters[4], &initial_gen, sizeof(int), cudaMemcpyHostToDevice), "Init Far Gen");

    // Barrier Array
    unsigned long long* d_barrier_val;
    check_cuda(cudaMalloc((void**)&d_barrier_val, sizeof(unsigned long long)), "Malloc Barrier");
    check_cuda(cudaMemset((void*)d_barrier_val, 0, sizeof(unsigned long long)), "Init Barrier");

    // Initialization
    int init_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_buffers<<<init_blocks, BLOCK_SIZE>>>(d_dist, n, source);
    check_cuda(cudaDeviceSynchronize(), "Init Buffers");

    // Setup Source
    check_cuda(cudaMemcpy(d_curr, &source, sizeof(int), cudaMemcpyHostToDevice), "Copy Source");
    int h_curr_size = 1;
    check_cuda(cudaMemcpy(&d_counters[0], &h_curr_size, sizeof(int), cudaMemcpyHostToDevice), "Set Source Size");

    auto compute_start = clock::now();

    // --- SAFE PERSISTENT LAUNCH CONFIG ---
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev_id);
    
    int num_sms = props.multiProcessorCount;
    int max_blocks_per_sm = 0;
    
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, sssp_persistent_kernel, BLOCK_SIZE, 0), "Occupancy Calc");
        
    // SAFETY: Use 85% of theoretical max to prevent system deadlocks/TDR
    int num_blocks = (int)((num_sms * max_blocks_per_sm) * 0.85);
    if (num_blocks < 1) num_blocks = 1;
    
    // Launch ONCE
    sssp_persistent_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_graph, d_dist, 
        d_curr, d_next_near, d_next_far, 
        d_counters, delta, 
        d_in_near_map, d_in_far_map,
        d_barrier_val, num_blocks
    );
    check_cuda(cudaGetLastError(), "Persistent Kernel Launch");
    check_cuda(cudaDeviceSynchronize(), "Persistent Kernel Sync");

    auto compute_end = clock::now();

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
    cudaFree((void*)d_barrier_val);
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