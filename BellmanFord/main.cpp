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
double cudaBellmanFord(Edge* edges, int* distances, int n, int m);

int main(int argc, char** argv)
{
    std::string input_filename;
    bool verbose = true;
    // Read command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:v")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
            break;
        case 'v':
            verbose = false;
            break;
        }
    }
    if (empty(input_filename) ) {
        std::cout << "Error with filename" << '\n';
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


    cudaTime = std::min(cudaTime, cudaBellmanFord(edges, distances, n, m));
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
