import random

random.seed(42) 

def write_graph(filename, n, edges):
    """Write graph in format n m source, then u v w for each edge"""
    m = len(edges)
    with open(filename, "w") as f:
        f.write(f"{n} {m} 0\n")
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")
    print(f"Created {filename}: {n} vertices, {m} edges")


def make_random_graph(n, m):
    edges = []
    
    vertices = list(range(n))
    random.shuffle(vertices)
    for i in range(n - 1):
        u, v = vertices[i], vertices[i + 1]
        w = random.randint(1, 100)
        edges.append((u, v, w))
    remaining = m - (n - 1)
    
    for _ in range(remaining):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        w = random.randint(1, 100)
        edges.append((u, v, w))
    
    return edges


def make_chain_graph(n):
    """chain like: 0 -> 1 -> 2 -> ... -> n-1 (worst case)"""
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1, random.randint(1, 10)))
    return edges


def make_star_graph(n):
    """Star like structure vertex 0 connects to all others (best case for frontier approach)"""
    edges = []
    for i in range(1, n):
        edges.append((0, i, random.randint(1, 100)))
    return edges


def make_cluster_graph(n, m, num_clusters):
    """Dense clusters with sparse inner cluster edges"""
    edges = []
    cluster_size = n // num_clusters
    
    intra_edges = int(m * 0.8)  # 80% within clusters
    edges_per_cluster = intra_edges // num_clusters
    
    for c in range(num_clusters):
        start = c * cluster_size
        end = start + cluster_size
        for _ in range(edges_per_cluster):
            u = random.randint(start, end - 1)
            v = random.randint(start, end - 1)
            w = random.randint(1, 50)
            edges.append((u, v, w))
    
    inter_edges = m - len(edges)
    for _ in range(inter_edges):
        c1 = random.randint(0, num_clusters - 1)
        c2 = random.randint(0, num_clusters - 1)
        while c2 == c1:
            c2 = random.randint(0, num_clusters - 1)
        u = random.randint(c1 * cluster_size, (c1 + 1) * cluster_size - 1)
        v = random.randint(c2 * cluster_size, (c2 + 1) * cluster_size - 1)
        w = random.randint(50, 200) 
        edges.append((u, v, w))
    
    return edges


def make_grid_graph(side):
    """2D grid graph (like a road network)"""
    n = side * side
    edges = []
    
    for i in range(side):
        for j in range(side):
            v = i * side + j
            if j < side - 1:
                edges.append((v, v + 1, random.randint(1, 50)))
                edges.append((v + 1, v, random.randint(1, 50)))
            if i < side - 1:
                edges.append((v, v + side, random.randint(1, 50)))
                edges.append((v + side, v, random.randint(1, 50)))
    
    return n, edges



# density scaling (fixed 10K vertices)

density_configs = [
    ("density_sparse_10000", 10000, 20000),      # 2 edges/vertex
    ("density_medium_10000", 10000, 100000),     # 10 edges/vertex
    ("density_dense_10000", 10000, 500000),      # 50 edges/vertex
    ("density_very_dense_10000", 10000, 1000000) # 100 edges/vertex
]

for name, n, m in density_configs:
    edges = make_random_graph(n, m)
    write_graph(f"{name}.txt", n, edges)


# size scaling (fixed 10 edges/vertex ratio)

scale_configs = [
    ("scale_1000", 1000, 10000),
    ("scale_5000", 5000, 50000),
    ("scale_10000", 10000, 100000),
    ("scale_50000", 50000, 500000),
    ("scale_100000", 100000, 1000000),
    ("scale_500000", 500000, 5000000)
]

for name, n, m in scale_configs:
    edges = make_random_graph(n, m)
    write_graph(f"{name}.txt", n, edges)


#  different structure types (50K vertices)

# random
edges = make_random_graph(50000, 500000)
write_graph("structure_random_50000.txt", 50000, edges)

# chain (worst case)
edges = make_chain_graph(50000)
write_graph("structure_chain_50000.txt", 50000, edges)

# star (best for frontier)
edges = make_star_graph(50000)
write_graph("structure_star_50000.txt", 50000, edges)

# cluster
edges = make_cluster_graph(50000, 500000, 10)
write_graph("structure_cluster_50000.txt", 50000, edges)

# road network like
side = 224  # = 50176 vertices
n, edges = make_grid_graph(side)
write_graph("structure_grid_50176.txt", n, edges)


#  real world graphs
# road network (sparse, grid like)
edges = make_random_graph(100000, 300000)
write_graph("realworld_road_100000.txt", 100000, edges)


