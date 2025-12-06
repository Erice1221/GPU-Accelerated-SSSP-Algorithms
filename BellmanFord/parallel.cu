#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>



#include "CycleTimer.h"

#define BLOCK_SIZE 256

struct Edge {
    int u, v;
    int w;
};

__global__ void iterateEdge(Edge* edges, int* distances, int m, int* isUpdated, int flagVal, int edgesPerThread) {
    // shared flag to see if any threads in the block have made an update
    __shared__ int blockUpdated;

    if (threadIdx.x == 0) {
        blockUpdated = 0;
    }
    __syncthreads();
    
    int baseIdx = (blockIdx.x * blockDim.x + threadIdx.x) * edgesPerThread;
    
    //each thread processes multiple concecutive edges
    for (int i = 0; i < edgesPerThread; i++) {
        int idx = baseIdx + i;
        if (idx < m) {
            Edge e = edges[idx];
            int distU = distances[e.u];
            if (distU != INT_MAX) {
                int dist = distU + e.w;
                int lDist = atomicMin(&distances[e.v], dist);
                if (dist < lDist) {
                    blockUpdated = 1;
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockUpdated) {
        *isUpdated = flagVal;
    }
}

__global__ void resetFlag(int* flag) {
    *flag = 0;
}



double cudaBellmanFordEdge(Edge* edges, int* distances,int n, int m)
{
    Edge* cEdges;
    int *cDistances;
    int *cIsUpdated;
    int isUpdated;

    // dynamic number of edges per thread depending on graph size
    int edgesPerThread;
    if (m >= 5000000) {        //5m edges
        edgesPerThread = 2;
    } else if (m >= 500000) {  //500k edges
        edgesPerThread = 4;
    } else if (m >= 50000) {   //50k edges
        edgesPerThread = 16;
    } else {  
        edgesPerThread = 32;
    }

    //allocate/copy gpu memory
    cudaMalloc(&cEdges, sizeof(Edge) * m);
    cudaMalloc(&cDistances, sizeof(int) * n);
    cudaMalloc(&cIsUpdated, sizeof(int));

    cudaMemcpy(cEdges, edges, sizeof(Edge) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(cDistances, distances, sizeof(int) * n, cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(cIsUpdated, &zero, sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int totalThreads = (m + edgesPerThread - 1) / edgesPerThread;
    dim3 grid((totalThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // main bellman ford loop with atmost n-1 iterations, launch kernels which relax edges in parallel
    for (int i = 0; i < n - 1; i++) {
        int currentFlag = i + 1;
        
        iterateEdge<<<grid, BLOCK_SIZE>>>(cEdges, cDistances, m, cIsUpdated, currentFlag, edgesPerThread);
        
        cudaMemcpy(&isUpdated, cIsUpdated, sizeof(int), cudaMemcpyDeviceToHost);

        if (isUpdated != currentFlag) { 
            break;
        }
    }
    
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(distances, cDistances, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(cEdges);
    cudaFree(cDistances);
    cudaFree(cIsUpdated);
    return overallDuration;
}
