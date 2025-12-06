#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define BLOCK_SIZE 256

struct Edge {
    int u, v;
    int w;
};

// thread computes the out degree of one frontier vertex
__global__ void computeDegrees(const int* rowPtr, const int* frontierCurr, int frontierSizeCurr, int* degrees)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSizeCurr) {
        return;
    }

    int u = frontierCurr[tid];
    degrees[tid] = rowPtr[u+1] - rowPtr[u];
}


// one thread per frontier edge
__global__ void relaxFrontierEdges(const Edge* edges, const int* rowPtr, int* distances, const int* frontierCurr, const int* edgeOffsets, int frontierSizeCurr, int totalFrontierEdges, int* frontierNext, int* frontierSizeNext)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalFrontierEdges) {
        return;
    } 
    // binary search to find which frontier vertex the edge belongs to
    int lo = 0;
    int hi = frontierSizeCurr;  

    while (lo+1 < hi) {
        int mid = (lo + hi) >> 1;
        if (tid < edgeOffsets[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // lo is now frontier vertex index
    int adjEdge = tid - edgeOffsets[lo];

    int u = frontierCurr[lo];
    int distU = distances[u];
    if (distU == INT_MAX) {
        return;
    }
    int eIdx = rowPtr[u] + adjEdge;

    Edge e = edges[eIdx];
    int v = e.v;

    int newDist = distU + e.w;
    int oldDist = atomicMin(&distances[v], newDist);

    if (newDist < oldDist) {
        int outPos = atomicAdd(frontierSizeNext, 1);
        frontierNext[outPos] = v; 
    }
}




double cudaBellmanFordFrontier(Edge* edges, int* distances,int n, int m, int source)
{
    // sort edges by source vertex for CSR adjacency list
    std::vector<Edge> sortedEdges(edges, edges + m);
    std::sort(sortedEdges.begin(), sortedEdges.end(),
        [](const Edge& a, const Edge& b) {
            return a.u < b.u;
    });

    std::vector<int> rowPtr(n + 1, 0);
    for (int i=0; i<m; i++) {
        int u = sortedEdges[i].u;
        rowPtr[u+1]++;
    }
    // prefix sum
    for (int i=0; i<n; i++) {
        rowPtr[i+1] += rowPtr[i];
    }

    Edge* cEdges;
    int* cDistances;
    int* cRowPtr;
    int* cFrontierCurr;
    int* cFrontierNext;
    int* cFrontierSizeNext;
    int* cDegrees;
    int* cEdgeOffsets;

    //allocate/copy gpu memory
    cudaMalloc(&cEdges, sizeof(Edge) * m);
    cudaMalloc(&cDistances, sizeof(int) * n);
    cudaMemcpy(cEdges, sortedEdges.data(), sizeof(Edge)*m, cudaMemcpyHostToDevice);
    cudaMemcpy(cDistances, distances, sizeof(int) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&cRowPtr, sizeof(int) * (n+1));
    cudaMemcpy(cRowPtr, rowPtr.data(), sizeof(int) * (n+1), cudaMemcpyHostToDevice);

    //frontier can contain all vertices(worst case)
    cudaMalloc(&cFrontierCurr, sizeof(int)*m);
    cudaMalloc(&cFrontierNext, sizeof(int)*m);
    cudaMalloc(&cFrontierSizeNext, sizeof(int));

    // for getting edge offsets each iteration
    cudaMalloc(&cDegrees, sizeof(int) * m);
    cudaMalloc(&cEdgeOffsets, sizeof(int) * (m+1));
    std::vector<int> Degrees(m);
    std::vector<int> EdgeOffsets(m+1);

    // start with just the source in the frontier
    int FrontierSizeCurr = 1;
    cudaMemcpy(cFrontierCurr, &source, sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int FrontierSizeNext;

    // main loop with at most n-1 iterations
    for (int i=0; i<n-1 && FrontierSizeCurr>0; i++) {
        // reset next frontier size
        int zero = 0;
        cudaMemcpy(cFrontierSizeNext, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // first compute out degree of each frontier vertex
        dim3 block(BLOCK_SIZE);
        dim3 grid((FrontierSizeCurr + BLOCK_SIZE - 1) / BLOCK_SIZE);

        computeDegrees<<<grid, block>>>(cRowPtr, cFrontierCurr, FrontierSizeCurr, cDegrees);
        cudaDeviceSynchronize();

        // copy degrees back to CPU for prefix sum
        cudaMemcpy(Degrees.data(), cDegrees, FrontierSizeCurr * sizeof(int), cudaMemcpyDeviceToHost);

        // prefix sum to get edge offsets for binary search
        EdgeOffsets[0] = 0;
        for (int j = 0; j < FrontierSizeCurr; j++) {
            EdgeOffsets[j+1] = EdgeOffsets[j] + Degrees[j];
        }
        int totalFrontierEdges = EdgeOffsets[FrontierSizeCurr];

        if (totalFrontierEdges == 0) {
            break;
        }
        cudaMemcpy(cEdgeOffsets, EdgeOffsets.data(),(FrontierSizeCurr + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // next relax all frontier edges in parallel (one thread per edge)
        dim3 grid2((totalFrontierEdges + BLOCK_SIZE - 1) / BLOCK_SIZE);
        relaxFrontierEdges<<<grid2, block>>>(cEdges, cRowPtr, cDistances,cFrontierCurr, cEdgeOffsets,FrontierSizeCurr, totalFrontierEdges,cFrontierNext, cFrontierSizeNext);
        cudaDeviceSynchronize();

        cudaMemcpy(&FrontierSizeNext, cFrontierSizeNext, sizeof(int), cudaMemcpyDeviceToHost);

        if (FrontierSizeNext == 0) {
            break;
        }

        // swap frontiers for next iteration
        std::swap(cFrontierCurr, cFrontierNext);
        FrontierSizeCurr = FrontierSizeNext;
    }
    
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(distances, cDistances, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(cEdges);
    cudaFree(cDistances);
    cudaFree(cRowPtr);
    cudaFree(cFrontierCurr);
    cudaFree(cFrontierNext);
    cudaFree(cFrontierSizeNext);
    cudaFree(cDegrees);
    cudaFree(cEdgeOffsets);
    return overallDuration;
}
