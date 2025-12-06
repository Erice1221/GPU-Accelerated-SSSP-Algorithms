#!/bin/bash

make sequential
make parallel

OUTPUT_FILE="test_results.txt"

for graph_file in input_graphs/*.txt; do
    echo "Testing: $graph_file"
    
    # Run sequential version
    seq_output=$(./sequential -f "$graph_file" 2>&1)
    seq_time=$(echo "$seq_output" | grep "Computation time" | awk '{print $4}')
    seq_distances=$(echo "$seq_output" | grep -A 1000 "v,distance")
    
    # Run parallel version
    par_output=$(./parallel -f "$graph_file" -m 'F' 2>&1)
    par_time=$(echo "$par_output" | grep "GPU_time" | awk '{print $2}')
    par_distances=$(echo "$par_output" | grep -A 1000 "v,distance")
    
    # Compare outputs
    if [ "$seq_distances" == "$par_distances" ]; then
        match="MATCH"
    else
        match="MISMATCH"
    fi

    speedup=$(echo "scale=2; $seq_time / $par_time" | bc)
    
    # write results to file
    echo "" >> $OUTPUT_FILE
    echo "Test file: $graph_file" >> $OUTPUT_FILE
    echo "Result: $match" >> $OUTPUT_FILE
    echo "Sequential time: $seq_time s" >> $OUTPUT_FILE
    echo "Parallel time: $par_time s" >> $OUTPUT_FILE
    echo "Speedup: ${speedup}x" >> $OUTPUT_FILE
    
    # print to console
    echo "  Result: $match"
    echo "  Sequential: $seq_time s"
    echo "  Parallel: $par_time s"
    echo "  Speedup: ${speedup}x"
    echo ""
done
