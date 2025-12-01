/**
 * @file main.cpp
 * @brief Entry point for GPU-accelerated SSSP algorithms.
 * @author Daniel Zhao
 * @date 2025-12-01
 */
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "graph.hpp"
#include "reader.hpp"
#include "dijkstra.hpp"
#include "delta.hpp"

/**
 * @brief Helper function to print usage information.
 * @param program_name The name of the executable.
 */
static void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << "-f <file_name> -m <method_id> [-s <source_index> -t <is_timed>]"
              << "  method_id = 0 : CPU-based Dijkstra\n"
              << "  method_id = 1 : GPU-based Delta-Stepping\n";
}

int main(int argc, char** argv) {
    std::string file_name;
    int method_id = -1;
    Graph::index_t source = 0;
    bool isTimed = false;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            file_name = argv[++i];
        }
        else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            method_id = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            source = static_cast<Graph::index_t>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            isTimed = (std::atoi(argv[++i]) != 0);
        }
        else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (file_name.empty() || method_id < 0) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        using clock = std::chrono::high_resolution_clock;
        using seconds = std::chrono::duration<double>;

        // Read the graph from file
        auto init_start = clock::now();
        Graph graph = read_graph_auto(file_name, true, 1.0f);
        auto init_end = clock::now();
        double init_time = seconds(init_end - init_start).count();
        if (isTimed) {
            std::cerr << "Initialization time: " << init_time << " s\n";
        }

        if (graph.num_vertices() == 0) {
            std::cerr << "Graph is empty.\n";
            return 1;
        }
        if (source < 0 || source >= graph.num_vertices()) {
            std::cerr << "Source vertex out of bounds.\n";
            std::cerr << "Vertex range: [0, " << graph.num_vertices() - 1 << "]\n";
            std::cerr << "Provided source: " << source << "\n";
            return 1;
        }

        // Execute the selected SSSP algorithm
        std::vector<Graph::weight_t> distances;
        if (method_id == 0) {
            double cpu_time = 0.0;
            double* cpu_time_ptr = isTimed ? &cpu_time : nullptr;
            distances = dijkstra_cpu(graph, source, nullptr, cpu_time_ptr);
            if (isTimed) {
                std::cerr << "Dijkstra CPU Compute time: " << cpu_time << " s\n";
            }
        }
        else if (method_id == 1) {
            const float delta = 1.0f; // Example delta value; can be parameterized
            double gpu_init_time = 0.0;
            double* gpu_init_time_ptr = isTimed ? &gpu_init_time : nullptr;
            double gpu_compute_time = 0.0;
            double* gpu_compute_time_ptr = isTimed ? &gpu_compute_time : nullptr;
            distances = delta_stepping_sssp(graph, source, delta, gpu_init_time_ptr, gpu_compute_time_ptr);

            if (isTimed) {
                std::cerr << "Additional GPU Initialization time: " << gpu_init_time << " s\n";
                std::cerr << "GPU Computation time: " << gpu_compute_time << " s\n";
            }
        }
        else {
            std::cerr << "Invalid method_id: " << method_id << "\n";
            print_usage(argv[0]);
            return 1;
        }

        for (Graph::index_t v = 0; v < graph.num_vertices(); ++v) {
            std::cout << v << " " << distances[static_cast<std::size_t>(v)] << "\n";
        }
    }
    catch(const std::exception& e) {
        std::cerr << "Error during execution: " << e.what() << "\n";
        return 1;
    }
    return 0;
}