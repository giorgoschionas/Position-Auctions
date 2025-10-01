"""
Visualization functions for budgeted bipartite matching simulations.

This module contains all plotting functions for analyzing algorithm performance,
including social welfare plots, approximation ratio, and theoretical
bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# Create figures directory in project root if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_distribution_pdfs(dists, dist_labels, x_range=(0, 10), n_points=1000):
    """
    Plot probability density functions for valuation distributions.

    Parameters:
    -----------
    dists : list
        List of scipy.stats distribution objects
    dist_labels : list
        Labels for each distribution
    x_range : tuple
        (min, max) values for x-axis
    n_points : int
        Number of points to plot
    """
    x = np.linspace(x_range[0], x_range[1], n_points)

    plt.figure(figsize=(8, 6))
    for i, dist in enumerate(dists):
        y = dist.pdf(x)
        plt.plot(x, y, label=dist_labels[i])

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Probability Density Functions')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "distributions.pdf"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_social_welfare(Bs, means, stds, opts, dist_labels, cost_dist_labels, colors):
    """
    Plot social welfare (greedy vs optimal) as a function of budget.

    Parameters:
    -----------
    Bs : np.ndarray
        Budget values
    means : np.ndarray
        Mean greedy values, shape (len(Bs), len(dists), len(cost_dists))
    stds : np.ndarray
        Standard deviation of greedy values
    opts : np.ndarray
        Optimal values
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    colors : list
        Colors for each distribution
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in zip(range(len(dist_labels)), dist_labels):
            plt.figure(figsize=(8, 6))
            plt.plot(Bs, means[:, i, k], lw=2, label='Greedy', color=colors[i])
            plt.plot(Bs, opts[:, i, k], lw=2, label='Optimal', linestyle='--', color='black')
            plt.fill_between(Bs, means[:, i, k] - stds[:, i, k],
                           means[:, i, k] + stds[:, i, k], alpha=0.5, color=colors[i])

            plt.yscale('log')
            plt.xlabel('Block Size (B)', fontsize=12)
            plt.ylabel('Social Welfare', fontsize=12)
            plt.title(f'Social Welfare vs Block Size\n({label} valuations, {cost_label} costs)', fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(FIGURES_DIR, f"SW-{label}-{cost_label}.pdf"), dpi=300, bbox_inches='tight')
            plt.show()


def plot_approximation_ratio(Bs, approxs_means, approxs_std, dist_labels, cost_dist_labels, colors):
    """
    Plot approximation ratio (OPT/SOL) as a function of budget.

    Parameters:
    -----------
    Bs : np.ndarray
        Budget values
    approxs_means : np.ndarray
        Mean approximation ratios, shape (len(Bs), len(dists), len(cost_dists))
    approxs_std : np.ndarray
        Standard deviation of approximation ratios
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    colors : list
        Colors for each distribution
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in zip(range(len(dist_labels)), dist_labels):
            plt.figure(figsize=(8, 6))
            plt.plot(Bs, approxs_means[:, i, k], lw=2, label='Approximation ratio', color=colors[i])
            plt.fill_between(Bs, approxs_means[:, i, k] - approxs_std[:, i, k],
                           approxs_means[:, i, k] + approxs_std[:, i, k], alpha=0.5, color=colors[i])

            plt.xlabel('Block Size (B)', fontsize=12)
            plt.ylabel('OPT/SOL', fontsize=12)
            plt.title(f'OPT/SOL vs Block Size\n({label} valuations, {cost_label} costs)', fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(FIGURES_DIR, f"Approx-{label}-{cost_label}.pdf"), dpi=300, bbox_inches='tight')
            plt.show()


def plot_approximation_histogram(approxs, dist_labels, cost_dist_labels, theoretical_bound=None):
    """
    Plot histogram of approximation ratios across all trials.

    This visualization is crucial for theoretical analysis as it shows the
    full distribution of approximation ratios, not just means.

    Parameters:
    -----------
    approxs : np.ndarray
        Array of shape (T, len(Bs), len(dists), len(cost_dists))
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    theoretical_bound : float, optional
        Theoretical worst-case bound to display as vertical line
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in enumerate(dist_labels):
            # Flatten all approximation ratios for this (dist, cost) combination
            ratios = approxs[:, :, i, k].flatten()
            ratios = ratios[ratios > 0]  # Remove zeros

            if len(ratios) == 0:
                continue

            plt.figure(figsize=(8, 6))
            plt.hist(ratios, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

            # Add theoretical bound if provided
            if theoretical_bound is not None:
                plt.axvline(theoretical_bound, color='red', linestyle='--', linewidth=2,
                           label=f'Theoretical Bound = {theoretical_bound}')

            # Add mean and max lines
            plt.axvline(np.mean(ratios), color='green', linestyle='-', linewidth=2,
                       label=f'Mean = {np.mean(ratios):.3f}')
            plt.axvline(np.max(ratios), color='orange', linestyle='-', linewidth=2,
                       label=f'Max = {np.max(ratios):.3f}')

            plt.xlabel('Approximation Ratio (OPT/SOL)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Approximation Ratios\n({label} valuations, {cost_label} costs)',
                     fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(FIGURES_DIR, f"Hist-Approx-{label}-{cost_label}.pdf"), dpi=300, bbox_inches='tight')
            plt.show()


def plot_approximation_cdf(approxs, dist_labels, cost_dist_labels, theoretical_bound=None):
    """
    Plot empirical CDF of approximation ratios to show tail behavior.

    The CDF is particularly useful for identifying worst-case behavior and
    validating that empirical results stay within theoretical bounds.

    Parameters:
    -----------
    approxs : np.ndarray
        Array of shape (T, len(Bs), len(dists), len(cost_dists))
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    theoretical_bound : float, optional
        Theoretical worst-case bound to display as vertical line
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in enumerate(dist_labels):
            # Flatten all approximation ratios for this (dist, cost) combination
            ratios = approxs[:, :, i, k].flatten()
            ratios = ratios[ratios > 0]  # Remove zeros

            if len(ratios) == 0:
                continue

            # Sort for CDF
            sorted_ratios = np.sort(ratios)
            cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

            plt.figure(figsize=(8, 6))
            plt.plot(sorted_ratios, cdf, linewidth=2, color='steelblue')

            # Add theoretical bound if provided
            if theoretical_bound is not None:
                plt.axvline(theoretical_bound, color='red', linestyle='--', linewidth=2,
                           label=f'Theoretical Bound = {theoretical_bound}')

            # Highlight tail (e.g., 95th percentile)
            p95 = np.percentile(ratios, 95)
            plt.axvline(p95, color='orange', linestyle=':', linewidth=2,
                       label=f'95th percentile = {p95:.3f}')

            plt.xlabel('Approximation Ratio (OPT/SOL)', fontsize=12)
            plt.ylabel('Cumulative Probability', fontsize=12)
            plt.title(f'CDF of Approximation Ratios\n({label} valuations, {cost_label} costs)',
                     fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(FIGURES_DIR, f"CDF-Approx-{label}-{cost_label}.pdf"), dpi=300, bbox_inches='tight')
            plt.show()


def plot_ratio_vs_budget_normalized(approxs, Bs, costs_data, dist_labels, cost_dist_labels):
    """
    Scatter plot: Approximation ratio vs normalized budget (B/max_cost).

    This plot reveals the relationship between budget tightness and algorithm
    performance, showing when the greedy algorithm struggles most.

    Parameters:
    -----------
    approxs : np.ndarray
        Array of shape (T, len(Bs), len(dists), len(cost_dists))
    Bs : np.ndarray
        Array of budget values
    costs_data : dict
        Dictionary storing max_cost for each trial, indexed by (trial, B_idx, dist_idx, cost_idx)
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in enumerate(dist_labels):
            budget_ratios = []
            approx_ratios = []

            # Collect all (budget_ratio, approx_ratio) pairs
            for t in range(approxs.shape[0]):
                for j, B in enumerate(Bs):
                    if (t, j, i, k) in costs_data:
                        max_cost = costs_data[(t, j, i, k)]
                        budget_ratio = B / max_cost if max_cost > 0 else 0
                        approx_ratio = approxs[t, j, i, k]

                        if approx_ratio > 0 and budget_ratio > 0:
                            budget_ratios.append(budget_ratio)
                            approx_ratios.append(approx_ratio)

            if len(budget_ratios) == 0:
                continue

            plt.figure(figsize=(8, 6))
            plt.scatter(budget_ratios, approx_ratios, alpha=0.5, s=30, color='steelblue')

            # Add trend line (polynomial fit)
            if len(budget_ratios) > 1:
                z = np.polyfit(budget_ratios, approx_ratios, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(budget_ratios), max(budget_ratios), 100)
                plt.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend (degree 2)')

            plt.xlabel('Budget / Max Cost', fontsize=12)
            plt.ylabel('Approximation Ratio (OPT/SOL)', fontsize=12)
            plt.title(f'Ratio vs Budget Tightness\n({label} valuations, {cost_label} costs)',
                     fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(os.path.join(FIGURES_DIR, f"Scatter-Ratio-Budget-{label}-{cost_label}.pdf"), dpi=300, bbox_inches='tight')
            plt.show()


def print_approximation_summary(max_approxs, dist_labels, cost_dist_labels):
    """
    Print summary of maximum approximation ratios.

    Parameters:
    -----------
    max_approxs : np.ndarray
        Array of shape (len(Bs), len(dists), len(cost_dists))
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    """
    print("\n" + "="*70)
    print("MAXIMUM APPROXIMATION RATIOS (OPT/SOL) ACROSS ALL TRIALS")
    print("="*70)

    for k, cost_label in enumerate(cost_dist_labels):
        print(f"\n{cost_label} costs:")
        for i, label in enumerate(dist_labels):
            max_ratio = np.max(max_approxs[:, i, k])
            print(f"  {label:20s}: {max_ratio:.4f}")

    print("="*70)
