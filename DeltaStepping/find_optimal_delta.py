#!/usr/bin/env python3

import subprocess
import re
import argparse
import sys
import os

def run_sssp(executable, graph_file, delta):
    """
    Runs the SSSP executable with the given delta and returns the GPU computation time.
    """
    cmd = [executable, "-f", graph_file, "-m", "1", "-d", str(delta), "-t", "1"]
    try:
        # Run the command and capture stderr (where the timing info is printed)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stderr
        
        # Look for "GPU Computation time: X s"
        match = re.search(r"GPU Computation time:\s+([0-9.]+)\s+s", output)
        if match:
            return float(match.group(1))
        else:
            # print(f"[Warning] Could not parse time for delta={delta}")
            return None
    except subprocess.CalledProcessError:
        # print(f"[Error] Failed to run for delta={delta}")
        return None

def find_optimal_for_file(executable, graph_file, deltas):
    best_time = float('inf')
    best_delta = -1
    
    print(f"Benchmarking {graph_file}...")
    
    for delta in deltas:
        time = run_sssp(executable, graph_file, delta)
        if time is not None:
            if time < best_time:
                best_time = time
                best_delta = delta
                
    return best_delta, best_time

def main():
    parser = argparse.ArgumentParser(description="Find optimal delta for SSSP")
    parser.add_argument("inputs", nargs='+', help="Paths to graph files or directories")
    parser.add_argument("--executable", default="./build/sssp", help="Path to sssp executable")
    parser.add_argument("--log", help="Path to output log file")
    
    args = parser.parse_args()

    if not os.path.exists(args.executable):
        print(f"Executable not found: {args.executable}")
        print("Please build the project first (e.g., 'make').")
        sys.exit(1)

    # Define a range of delta values to test
    deltas = [
        1.0, 2.0, 5.0, 
        10.0, 20.0, 50.0, 
        100.0, 200.0, 500.0, 
        1000.0, 2000.0, 5000.0, 
        10000.0, 20000.0, 50000.0, 
        100000.0, 200000.0, 500000.0
    ]

    files_to_process = []
    for inp in args.inputs:
        if os.path.isfile(inp):
            files_to_process.append(inp)
        elif os.path.isdir(inp):
            for root, dirs, files in os.walk(inp):
                for file in files:
                    if file.endswith(".gr") or file.endswith(".txt"):
                        files_to_process.append(os.path.join(root, file))
    
    # Remove duplicates and sort
    files_to_process = sorted(list(set(files_to_process)))
    
    results = []
    
    print(f"Found {len(files_to_process)} files to benchmark.")
    print("-" * 50)
    
    for graph_file in files_to_process:
        best_delta, best_time = find_optimal_for_file(args.executable, graph_file, deltas)
        
        if best_delta != -1:
            print(f"  -> Best Delta: {best_delta:<10} Time: {best_time:.5f}s")
            results.append((graph_file, best_delta, best_time))
        else:
            print(f"  -> Failed to benchmark {graph_file}")

    print("-" * 50)

    if args.log:
        try:
            with open(args.log, "w") as f:
                f.write("File Path,Optimal Delta,Time (s)\n")
                for res in results:
                    f.write(f"{res[0]},{res[1]},{res[2]:.5f}\n")
            print(f"Results written to {args.log}")
        except IOError as e:
            print(f"Error writing to log file: {e}")

if __name__ == "__main__":
    main()
