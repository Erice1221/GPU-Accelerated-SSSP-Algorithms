#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <climits>

#include "CycleTimer.h"
struct Edge {
    int u, v;
    int w;
};
double cudaBellmanFordEdge(Edge* edges, int* distances, int n, int m);
double cudaBellmanFordFrontier(Edge* edges, int* distances, int n, int m, int source);

int main(int argc, char** argv)
{
    std::string input_filename;
    bool verbose = true;
    char parallel_mode = '\0';
    // Read command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:v:m:")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
            break;
        case 'v':
            verbose = false;
            break;
        case 'm':
            parallel_mode = *optarg;
            break;
        }
    }
    if (empty(input_filename) ) {
        std::cout << "Error with filename" << '\n';
        exit(EXIT_FAILURE);
    }
    if (parallel_mode != 'E' && parallel_mode != 'F') {
        std::cout << "Include Parallel mode -m E or -m F" << '\n';
        exit(EXIT_FAILURE);
    }
    
    std::ifstream fin(input_filename);

    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
    }


    // n is number of verticies, m is number of edges
    int n,m,source;
    fin >> n >> m >> source;
    Edge* edges = new Edge[m];

    for (int i=0; i<m;i++) {
        Edge e;
        fin >> e.u >> e.v >> e.w;
        edges[i] = e;
    }

    int* distances = new int[n];
    for (int i = 0; i < n; i++) {
        distances[i] = INT_MAX;
    }
    distances[source] = 0;
    

    double cudaTime = 50000.;
    if (parallel_mode == 'E') {
        cudaTime = std::min(cudaTime, cudaBellmanFordEdge(edges, distances, n, m));
    }
    if (parallel_mode == 'F') {
        cudaTime = std::min(cudaTime, cudaBellmanFordFrontier(edges, distances, n, m, source));
    }
    printf("GPU_time: %f s\n", cudaTime);
    if (verbose) {
        printf("v,distance\n");
        for (int i=0;i<n;i++) {
            printf("%d,%d\n",i,distances[i]);
        }
    }
    
    delete[] edges;
    delete[] distances;
    return 0;
}
