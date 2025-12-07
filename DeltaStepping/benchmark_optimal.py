#!/usr/bin/env python3

import csv
import subprocess
import re
import sys
import os

def parse_output(output):
    data = {}
    # Initialization time: 1.44733 s
    m_init = re.search(r"Initialization time:\s+([0-9.]+)\s+s", output)
    if m_init:
        data['init_time'] = float(m_init.group(1))
    
    # Dijkstra CPU Compute time: 0.134641 s
    m_cpu = re.search(r"Dijkstra CPU Compute time:\s+([0-9.]+)\s+s", output)
    if m_cpu:
        data['cpu_time'] = float(m_cpu.group(1))
        
    # Additional GPU Initialization time: 2.25857 s
    m_gpu_init = re.search(r"Additional GPU Initialization time:\s+([0-9.]+)\s+s", output)
    if m_gpu_init:
        data['gpu_init_time'] = float(m_gpu_init.group(1))
        
    # GPU Computation time: 1.32536 s
    m_gpu_comp = re.search(r"GPU Computation time:\s+([0-9.]+)\s+s", output)
    if m_gpu_comp:
        data['gpu_comp_time'] = float(m_gpu_comp.group(1))
        
    return data

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stderr # Output is on stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        return None

def main():
    input_log = "optimal_deltas.log"
    output_log = "final_benchmark_results.csv"
    executable = "./build/sssp"
    
    if not os.path.exists(input_log):
        print(f"{input_log} not found.")
        return

    tasks = []
    with open(input_log, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader) # Skip header
        except StopIteration:
            pass # Empty file
            
        for row in reader:
            if len(row) >= 2:
                tasks.append((row[0], float(row[1])))
    
    results = []
    print(f"Benchmarking {len(tasks)} graphs...")
    
    for file_path, delta in tasks:
        print(f"Processing {file_path} with delta={delta}...")
        
        # Run CPU
        cmd_cpu = [executable, "-f", file_path, "-m", "0", "-t", "1"]
        out_cpu = run_command(cmd_cpu)
        if not out_cpu: continue
        data_cpu = parse_output(out_cpu)
        
        # Run GPU
        cmd_gpu = [executable, "-f", file_path, "-m", "1", "-d", str(delta), "-t", "1"]
        out_gpu = run_command(cmd_gpu)
        if not out_gpu: continue
        data_gpu = parse_output(out_gpu)
        
        # Combine results
        # CPU Init Time (from CPU run)
        cpu_init = data_cpu.get('init_time', 0.0)
        # Dijkstra Compute Time
        dijkstra_time = data_cpu.get('cpu_time', 0.0)
        
        # GPU Init Time (Graph load from GPU run - should be similar to CPU init but let's record it)
        gpu_load_time = data_gpu.get('init_time', 0.0)
        # GPU Additional Init
        gpu_add_init = data_gpu.get('gpu_init_time', 0.0)
        # GPU Compute
        gpu_comp = data_gpu.get('gpu_comp_time', 0.0)
        
        results.append({
            'file': file_path,
            'delta': delta,
            'cpu_init': cpu_init,
            'dijkstra_time': dijkstra_time,
            'gpu_load_time': gpu_load_time,
            'gpu_add_init': gpu_add_init,
            'gpu_comp': gpu_comp
        })

    # Write results
    with open(output_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File Path", "Delta Value", 
            "CPU Initialization Time", "Dijkstra Computation Time", 
            "GPU Initialization Time", "GPU Additional Initialization Time", "GPU Computation Time"
        ])
        for r in results:
            writer.writerow([
                r['file'], r['delta'],
                r['cpu_init'], r['dijkstra_time'],
                r['gpu_load_time'], r['gpu_add_init'], r['gpu_comp']
            ])
            
    print(f"Results saved to {output_log}")

if __name__ == "__main__":
    main()
