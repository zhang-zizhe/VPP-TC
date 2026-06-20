#!/usr/bin/env python3
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
import functions as functions_collision

# # Input data (fixed or randomized)
# q = np.random.rand(7).astype(np.float32)
# q_dot = np.random.rand(7).astype(np.float32)

# # Warm-up run (optional)
# functions_collision.compute_gamma(q, q_dot)

# # Run 100 times and time it
# num_runs = 100
# total_time = 0.0
# results = []

# for _ in range(num_runs):
#     start = time.perf_counter()
#     gamma_val, _ = functions_collision.compute_gamma_and_grad(q, q_dot, threshold=10.1)
#     end = time.perf_counter()
    
#     elapsed = end - start
#     total_time += elapsed
#     results.append(elapsed)

# # Report
# avg_time_us = (total_time / num_runs) * 1e6  # convert to microseconds
# print(f"Last gamma_val: {gamma_val}")
# print(f"Average execution time over {num_runs} runs: {avg_time_us:.2f} µs")

# q = np.random.rand(7).astype(np.float32)
# q_dot = np.random.rand(7).astype(np.float32)

# Warm-up (optional, for fair timing on JIT/cached functions)
# functions_collision.compute_gamma(q, q_dot)
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

while True:
    # Sample q uniformly within [q_min, q_max]
    q = np.random.uniform(low=q_min, high=q_max).astype(np.float32)

    # Sample q_dot uniformly within [-qd_lim, +qd_lim]
    q_dot = np.random.uniform(low=-qd_lim, high=qd_lim).astype(np.float32)
    # Start timer
    start = time.perf_counter()

    # Call the function
    gamma_val = functions_collision.compute_gamma(q, q_dot)

    # Stop timer
    end = time.perf_counter()

    # Report result
    # print(f"gamma_val: {gamma_val}")
    print(f"Execution time: {(end - start) * 1e3:.2f} ms")
