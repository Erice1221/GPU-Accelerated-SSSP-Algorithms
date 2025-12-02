#include <iostream>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <string>
#include <iomanip>
#include <climits>
#include <vector>
struct Edge {
    int u, v;
    int w;
};

bool bellmanFord(const std::vector<Edge>& graph, std::vector<int>& distances, int n) {

    //At most n-1 iterations
    for (int i=0;i < n-1; i++) {
        bool isUpdated = false;
        for (const auto& edge: graph ) {
            //check for shorter path
            if (distances[edge.u] != INT_MAX) {
                int dist = distances[edge.u] + edge.w;
                if (dist < distances[edge.v]) {
                    distances[edge.v] = dist;
                    isUpdated = true;
                }
            }  
        }
        if (!isUpdated) {
            break;
        }
    }

    //check for negative cycles
    // for (const auto& edge: graph) {
    //     if (distances[edge.u] != INT_MAX && distances[edge.v] > distances[edge.u] + edge.w) {
    //         //negative cycle found
    //         return true;
    //     }

    // }

    // no negative cycle
    return false;

}

int main(int argc, char *argv[]) {

    const auto init_start = std::chrono::steady_clock::now();

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
    std::vector<Edge> edges;
    edges.reserve(m);

    for (int i=0; i<m;i++) {
        Edge e;
        fin >> e.u >> e.v >> e.w;
        edges.push_back(e);
    }

    std::vector<int> distances(n,INT_MAX);
    distances[source] = 0;
  
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

   
    if (bellmanFord(edges,distances,n)) {
        std::cout << "Negative cycle found" << '\n';
        exit(1);
    }
  
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
    
    if (verbose) {
        std::cout << "v,distance" << '\n';
        for (int i=0;i<n;i++) {
            std::cout << i << "," << distances[i] << '\n';
        }
    }
    return 0;
}