# Milestone Report for GPU-Accelerated Single Source Shortest Path (SSSP) Algorithms

## URL

The project repository can be found at: [https://github.com/Erice1221/GPU-Accelerated-SSSP-Algorithms](https://github.com/Erice1221/GPU-Accelerated-SSSP-Algorithms)

## Work Completed So Far

Daniel: So far, I have implemented a single-threaded version of Dijkstra's algorithm in C++ as a baseline for performance comparison. I also created a basic CUDA implementation of Delta-Stepping SSSP algorithm. The CUDA implementation currently is able to preduce correct shortest path results on graphs, but it is not yet optimized for performance. I was expected to have some additional optimizations done by this milestone, but I ran into some unexpected challenges with memory management on the GPU, which slowed down my progress. I decided to focus on getting a correct implementation first before diving into optimizations, and would aim for a better optimized delta-stepping implementation before the final submission. In addition, I have also setup the program structure and testing framework to facilitate further development and benchmarking. Further work to help Eric to integrate his work on Bellman-Ford algorithms would also be required. 

Eric: 

## Preliminary Results

Daniel: The single-threaded Dijkstra's algorithm performs as expected on small to medium-sized graphs, providing correct shortest path results. The CUDA Delta-Stepping implementation also produces correct results, but its performance is currently suboptimal compared to the CPU version due to lack of optimizations. Potential areas for improvement include better memory access patterns, reducing thread divergence, and optimizing the bucket management in the Delta-Stepping algorithm.

Eric: 

## Concerns

Daniel: Currently, I don't have major concerns, but I am aware that optimizing GPU algorithms can be quite challenging, especially with memory management and ensuring efficient parallelism. I plan to allocate sufficient time for testing and optimization in the coming weeks, and would contact the course staff if I encounter any significant roadblocks.

Eric: 

## Refined Schedule

Dec 1 - Dec 3:
- Daniel: Finalize an optimized implementation of Delta-Stepping SSSP algorithm on GPU.
- Eric: 

Dec 4 - Dec 6:
- Daniel: Begin integrating Bellman-Ford algorithm implementation on GPU. Starts testing and benchmarking both algorithms against the CPU baseline.
- Eric:

Dec 7: 
- Both: Final testing, benchmarking, and documentation. Prepare for submission.