# Bellman Ford

## Parallel (CUDA) version
1. compile using `make parallel`
2. Run with `./parallel -f PATH_TO_FILE -m PARALLEL_MODE` (-m E for edge parallel, -m F for frontier parallel) (example: ./parallel -f ./graph/test.txt -m F)

## Sequential (c++) version
1. compile using `make sequential`
2. Run with `./sequential -f PATH_TO_FILE` (example: ./sequential -f ./graph/test.txt)

Optional add `-v` to hide shortest path from source output (good for large graphs to see runtime)

## Input Graphs 
/input_graphs/ includes `make_test_graphs.py` which generates graphs that can be used on both versions (all *.txt files a run with checker.sh).

## Checker
Run `checker.sh` to run both sequential and parallel version and compare outputs