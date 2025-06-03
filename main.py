# main.py
import argparse
import time
import pandas as pd
import torch
import numpy as np


from safety_bounds import (
    gamma_model,
    online_search,
    online_gridsearch,
)

def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate safe joint position bounds for a set of (q, qd) samples"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="samples.csv",
        help="Input CSV containing 14 columns: q1…q7, qd1…qd7",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["search", "grid"],
        default="search",
        help="'search' = online_search (parallel bisection); "
             "'grid' = online_gridsearch (fixed step scan)",
    )
    return parser.parse_args()



def compute_bounds(q, qd, method, a_max_np):
    if method == "search":
        return online_search(
            q, qd, gamma_model, a_max_np,
            delta_t=0.02,
            tol=1e-5,
            threshold=2.5,
        )
    elif method == "grid":
        return online_gridsearch(
            q, qd, gamma_model, a_max_np,
            delta_t=0.02,
            step_size=0.1,
            threshold=2.5,
        )
    else:
        raise ValueError(f"Unknown method: {method}")



def main():
    args = get_args()

    df = pd.read_csv(args.csv)
    q_all  = torch.tensor(df.iloc[:, 0:7 ].values, dtype=torch.float32)
    qd_all = torch.tensor(df.iloc[:, 7:14].values, dtype=torch.float32)
    a_max_np = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)

    all_bounds = []
    total_time = 0.0

    for idx, (q, qd) in enumerate(zip(q_all, qd_all), 1):
        t0 = time.time()
        try:
            q_min, q_max = compute_bounds(q, qd, args.method, a_max_np)

            assert q_min.shape == (7,) and q_max.shape == (7,), \
                f"Returned shape mismatch at sample {idx}"

            all_bounds.append(np.ravel(np.column_stack((q_min, q_max))))
            total_time += time.time() - t0
        except Exception as e:
            print(f"sample {idx} skipped – {e}")

    if all_bounds:
        out_csv = f"estimated_bounds_{args.method}.csv"
        cols = [f"q{i}_{s}" for i in range(7) for s in ("min", "max")]
        pd.DataFrame(all_bounds, columns=cols).to_csv(out_csv, index=False)

        avg = total_time / len(all_bounds)
        print(f"{len(all_bounds)} samples processed, avg time {avg:.4f}s")
        print(f"Results saved to: {out_csv}")
    else:
        print("No valid samples to save.")

    for idx, (q, qd) in enumerate(zip(q_all, qd_all), 1):
        t0 = time.time()
        try:
            q_min, q_max = compute_bounds(q, qd, args.method, a_max_np)

            assert q_min.shape == (7,) and q_max.shape == (7,), \
                f"Returned shape mismatch at sample {idx}"

            print(f"Sample {idx}")
            print(f"  q_min: {q_min}, type = {type(q_min)}, shape = {q_min.shape}")
            print(f"  q_max: {q_max}, type = {type(q_max)}, shape = {q_max.shape}")

            all_bounds.append(np.ravel(np.column_stack((q_min, q_max))))
            total_time += time.time() - t0
        except Exception as e:
            print(f"Sample {idx} skipped – {e}")
if __name__ == "__main__":
    main()
