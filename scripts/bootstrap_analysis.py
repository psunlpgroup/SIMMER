#!/usr/bin/env python3
"""
Bootstrap confidence intervals and paired t-tests for plan evaluation results.

Computes:
1. Per-method bootstrap 95% CIs for key metrics
2. Pairwise paired t-tests between methods
3. Outputs a summary table suitable for the paper

Usage:
    python bootstrap_analysis.py --dirs ./results/gpt5:Vanilla ./results/gpt5.4_foresight:Foresight ./results/gpt5.4_react:ReAct
    python bootstrap_analysis.py --dirs ./results/gpt5:Vanilla ./results/gpt5.4_foresight:Foresight --metric total
"""

import argparse
import json
import os
import re
import sys
import pathlib
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from simmer.paths import ACTION_DEFS, RUNS_DIR
from simmer.state_machine import KitchenStateMachine, FailureType

ACTION_DEFS_PATH = str(ACTION_DEFS)


def evaluate_single_plan(folder: str, base_dir: str) -> dict:
    """Evaluate a single plan and return per-task failure counts."""
    env_path = f'{base_dir}/{folder}/env_{folder}.json'
    plan_path = f'{base_dir}/{folder}/plan_{folder}.txt'

    if not os.path.exists(env_path) or not os.path.exists(plan_path):
        return None
    if os.path.getsize(plan_path) == 0:
        return None

    # Check if plan contains an error
    with open(plan_path) as f:
        first_line = f.readline().strip()
    if first_line.startswith('Error:'):
        return None

    try:
        sm = KitchenStateMachine(env_path, ACTION_DEFS_PATH)
        sm.execute_plan_file(plan_path)
    except Exception:
        return None

    deduped = sm.deduplicated_failures()
    imm = sum(1 for d in deduped if d['failure_type'] != FailureType.LATENT)
    latent = sum(1 for d in deduped if d['failure_type'] == FailureType.LATENT)
    irreversible = sum(1 for d in deduped if not d['reversible'])
    total = imm + latent
    clean = 1 if total == 0 else 0

    return {
        'folder': folder,
        'total': total,
        'immediate': imm,
        'latent': latent,
        'irreversible': irreversible,
        'clean': clean,
    }


def collect_results(base_dir: str) -> list:
    """Collect per-task results for a method."""
    folders = sorted([f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f))])
    results = []
    for folder in folders:
        r = evaluate_single_plan(folder, base_dir)
        if r is not None:
            results.append(r)
    return results


def bootstrap_ci(values, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval for the mean."""
    values = np.array(values)
    n = len(values)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[np.random.randint(0, n, size=n)]
        boot_means[i] = sample.mean()
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return values.mean(), lower, upper


def bootstrap_ci_sum(values, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval for the sum."""
    values = np.array(values)
    n = len(values)
    boot_sums = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[np.random.randint(0, n, size=n)]
        boot_sums[i] = sample.sum()
    lower = np.percentile(boot_sums, (100 - ci) / 2)
    upper = np.percentile(boot_sums, 100 - (100 - ci) / 2)
    return values.sum(), lower, upper


def bootstrap_ci_rate(values, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval for a rate (proportion)."""
    values = np.array(values)
    n = len(values)
    boot_rates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[np.random.randint(0, n, size=n)]
        boot_rates[i] = sample.mean()
    lower = np.percentile(boot_rates, (100 - ci) / 2)
    upper = np.percentile(boot_rates, 100 - (100 - ci) / 2)
    return values.mean(), lower, upper


def paired_t_test(values_a, values_b):
    """Paired t-test on per-task differences. Returns (t_stat, p_value)."""
    a = np.array(values_a)
    b = np.array(values_b)
    # Align by task — use minimum length
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    t_stat, p_value = stats.ttest_rel(a, b)
    return t_stat, p_value


def paired_bootstrap_test(values_a, values_b, n_bootstrap=10000):
    """Bootstrap test for difference in means. Returns (mean_diff, ci_lower, ci_upper, p_value)."""
    a = np.array(values_a)
    b = np.array(values_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = a - b
    observed_diff = diffs.mean()

    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = diffs[np.random.randint(0, n, size=n)]
        boot_diffs[i] = sample.mean()

    lower = np.percentile(boot_diffs, 2.5)
    upper = np.percentile(boot_diffs, 97.5)
    # Two-sided p-value: proportion of bootstrap samples crossing zero
    if observed_diff > 0:
        p_value = (boot_diffs <= 0).sum() / n_bootstrap * 2
    else:
        p_value = (boot_diffs >= 0).sum() / n_bootstrap * 2
    p_value = min(p_value, 1.0)

    return observed_diff, lower, upper, p_value


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap CIs and paired tests for plan evaluation')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Method directories in format path:Label (e.g., ./results/gpt5:Vanilla)')
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Parse method directories
    methods = {}
    for entry in args.dirs:
        if ':' in entry:
            path, label = entry.rsplit(':', 1)
        else:
            path = entry
            label = Path(entry).name
        methods[label] = path

    # Collect results
    print("Collecting results...")
    all_results = {}
    for label, path in methods.items():
        results = collect_results(path)
        all_results[label] = results
        print(f"  {label}: {len(results)} tasks evaluated")

    # Metrics to analyze
    metrics = ['total', 'immediate', 'latent', 'irreversible', 'clean']

    # === Bootstrap CIs ===
    print(f"\n{'='*80}")
    print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS (n_bootstrap={args.n_bootstrap})")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Method':<25} {'Total':>18} {'Immediate':>18} {'Latent':>18} {'Irreversible':>18} {'Clean Rate':>18}")
    print("-" * 115)

    method_values = {}  # label -> metric -> list of per-task values
    for label, results in all_results.items():
        method_values[label] = {}
        row = f"{label:<25}"

        for metric in metrics:
            values = [r[metric] for r in results]
            method_values[label][metric] = values

            if metric == 'clean':
                mean, lower, upper = bootstrap_ci_rate(values, args.n_bootstrap)
                n = len(values)
                row += f" {mean*100:5.1f}% [{lower*100:.1f},{upper*100:.1f}]"
            else:
                total, lower, upper = bootstrap_ci_sum(values, args.n_bootstrap)
                row += f" {total:5.0f} [{lower:.0f},{upper:.0f}]  "

        print(row)

    # === Pairwise Paired Tests ===
    labels = list(all_results.keys())
    if len(labels) >= 2:
        print(f"\n{'='*80}")
        print(f"PAIRWISE PAIRED TESTS (metric: total failures per task)")
        print(f"{'='*80}")
        print(f"\n{'Comparison':<40} {'Mean Diff':>10} {'95% CI':>18} {'p-value':>10} {'t-test p':>10}")
        print("-" * 90)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                va = method_values[la]['total']
                vb = method_values[lb]['total']

                # Bootstrap paired test
                mean_diff, ci_lower, ci_upper, boot_p = paired_bootstrap_test(
                    va, vb, args.n_bootstrap)

                # Paired t-test
                t_stat, t_p = paired_t_test(va, vb)

                sig = "***" if boot_p < 0.001 else "**" if boot_p < 0.01 else "*" if boot_p < 0.05 else "ns"

                comp = f"{la} vs {lb}"
                print(f"{comp:<40} {mean_diff:>+10.2f} [{ci_lower:>+.2f},{ci_upper:>+.2f}]  {boot_p:>8.4f} {t_p:>8.4f}  {sig}")

        # Also test on immediate and latent separately
        for metric_name in ['immediate', 'latent', 'irreversible']:
            print(f"\n--- Pairwise tests on: {metric_name} failures ---")
            print(f"{'Comparison':<40} {'Mean Diff':>10} {'95% CI':>18} {'p-value':>10}")
            print("-" * 80)
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    la, lb = labels[i], labels[j]
                    va = method_values[la][metric_name]
                    vb = method_values[lb][metric_name]
                    mean_diff, ci_lower, ci_upper, boot_p = paired_bootstrap_test(
                        va, vb, args.n_bootstrap)
                    sig = "***" if boot_p < 0.001 else "**" if boot_p < 0.01 else "*" if boot_p < 0.05 else "ns"
                    comp = f"{la} vs {lb}"
                    print(f"{comp:<40} {mean_diff:>+10.2f} [{ci_lower:>+.2f},{ci_upper:>+.2f}]  {boot_p:>8.4f}  {sig}")


if __name__ == '__main__':
    main()
