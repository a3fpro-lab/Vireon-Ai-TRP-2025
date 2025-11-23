"""
Plotting stub for prereg results.

Expected inputs:
- results/**/logs.json or csv saved by scripts/main.py
- contains per-step returns, dt_eff, R_t, P_t, KL

This file will be filled in once runs exist.
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_logs(results_dir="results"):
    paths = glob.glob(os.path.join(results_dir, "**", "logs.json"), recursive=True)
    all_logs = []
    for p in paths:
        with open(p, "r") as f:
            all_logs.append((p, json.load(f)))
    return all_logs


def plot_returns(all_logs):
    plt.figure()
    for path, logs in all_logs:
        ret = logs.get("returns", [])
        if len(ret) == 0:
            continue
        plt.plot(ret, alpha=0.6, label=os.path.basename(os.path.dirname(path)))
    plt.title("Episodic Returns (per seed/run)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.show()


def main():
    logs = load_logs()
    if not logs:
        print("No logs found yet. Run scripts/main.py first.")
        return
    plot_returns(logs)


if __name__ == "__main__":
    main()
