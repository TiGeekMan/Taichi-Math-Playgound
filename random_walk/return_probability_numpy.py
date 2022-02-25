"""
Numpy version.
Return probability of a simple random walk on Z^d.
"""
import numpy as np
import time


start = time.perf_counter()
dim = 3
num_rounds = 1000000
max_steps = 100000
origin = np.zeros(dim, dtype=int)

I = np.eye(dim, dtype=int)
neighbours = np.append(I, -I, axis=0)

def get_random_direction():
    rnd = np.random.randint(0, 2*dim)
    return neighbours[rnd]

def do_experiments():
    success_count = 0
    for _ in range(num_rounds):
        pos = np.zeros(dim, dtype=int)
        for step in range(max_steps):
            pos += get_random_direction()
            if (pos == origin).all():
                success_count += 1
                break
    return success_count

success_count = do_experiments()
print(success_count/ num_rounds)
print(time.perf_counter() - start)