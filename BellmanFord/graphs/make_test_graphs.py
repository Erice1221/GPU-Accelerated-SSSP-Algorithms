import random

# n = 1000  # verticies
# m = 10000  # edges

# with open(f"test_{n}.txt", "w") as f:
#     f.write(f"{n} {m} 0\n")
#     for _ in range(m):
#         u = random.randint(0, n-1)
#         v = random.randint(0, n-1)
#         w = random.randint(1, 100)
#         f.write(f"{u} {v} {w}\n")

# with open(f"hard_{n}.txt", "w") as f:
#     f.write(f"{n} {m} 0\n")

#     for i in range(n-1):
#         f.write(f"{i} {i+1} 1\n")

#     for _ in range(m - (n-1)):
#         u = random.randint(0, n-1)
#         v = random.randint(0, n-1)
#         w = random.randint(1, 50)
#         f.write(f"{u} {v} {w}\n")

# n = 10000   
# m = 500000  

# # graph thats better for gpu source connects to many nodes 
# with open(f"gpu_{n}.txt", "w") as f:
#     f.write(f"{n} {m} 0\n")
#     for i in range(1, 1000):
#         f.write(f"0 {i} {random.randint(1, 10)}\n")

#     for _ in range(m - 999):
#         u = random.randint(0, n-1)
#         v = random.randint(0, n-1)
#         w = random.randint(1, 100)
#         f.write(f"{u} {v} {w}\n")

# n = 1000 

# with open("pure_chain.txt", "w") as f:
#     f.write(f"{n} {n-1} 0\n")
#     for i in range(n-1):
#         f.write(f"{i} {i+1} 1\n")


# n = 500000 
# m = 10000000

# with open("huge_500000.txt", "w") as f:
#     f.write(f"{n} {m} 0\n")
#     for _ in range(m):
#         u = random.randint(0, n-1)
#         v = random.randint(0, n-1)
#         w = random.randint(1, 100)
#         f.write(f"{u} {v} {w}\n")