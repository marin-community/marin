#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize plateau detection on various reward trajectories."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from marin.rl.curriculum import LessonStats, PerformanceStats, is_plateaued

sns.set_theme(style="darkgrid")
sns.set_palette("husl")


def generate_trajectories(n_steps: int = 200) -> dict[str, np.ndarray]:
    """Generate various reward trajectory patterns."""
    trajectories = {}
    rng = np.random.default_rng(seed=42)

    # 1. Linear growth then plateau (high)
    t = np.arange(n_steps)
    traj = np.clip(t / 50, -1, 0.8) + rng.normal(0, 0.05, n_steps)
    trajectories["Linear → Plateau (high)"] = traj

    # 2. Linear growth then plateau (low noise)
    traj = np.clip(t / 60, -1, 0.7) + rng.normal(0, 0.02, n_steps)
    trajectories["Linear → Plateau (low noise)"] = traj

    # 3. Fast growth then plateau
    traj = np.clip(t / 30, -1, 0.9)
    traj[100:] = 0.9 + rng.normal(0, 0.03, n_steps - 100)
    trajectories["Fast growth → Plateau"] = traj

    # 4. Sinusoidal meandering (no clear plateau)
    traj = 0.3 * np.sin(t / 15) + 0.01 * t + rng.normal(0, 0.05, n_steps)
    trajectories["Sinusoidal meandering"] = np.clip(traj, -1, 1)

    # 5. Damped oscillation converging
    traj = 0.8 * np.exp(-t / 40) * np.sin(t / 5) + 0.5
    traj += rng.normal(0, 0.03, n_steps)
    trajectories["Damped oscillation"] = np.clip(traj, -1, 1)

    # 6. Flat from start (no progress)
    traj = np.zeros(n_steps) + rng.normal(0, 0.05, n_steps)
    trajectories["Flat (no progress)"] = traj

    # 7. Flat at low value
    traj = -0.5 * np.ones(n_steps) + rng.normal(0, 0.03, n_steps)
    trajectories["Flat (low, stable)"] = traj

    # 8. Slow continuous growth (not plateaued)
    traj = -0.5 + 0.008 * t + rng.normal(0, 0.04, n_steps)
    trajectories["Slow continuous growth"] = np.clip(traj, -1, 1)

    # 9. Noisy plateau (high variance)
    traj = np.ones(n_steps) * 0.6
    traj[:50] = np.linspace(-0.3, 0.6, 50)
    traj += rng.normal(0, 0.15, n_steps)
    trajectories["Noisy plateau (high var)"] = np.clip(traj, -1, 1)

    # 10. Step function (sudden plateau)
    traj = -0.5 * np.ones(n_steps)
    traj[80:] = 0.7
    traj += rng.normal(0, 0.04, n_steps)
    trajectories["Step → Plateau"] = traj

    return trajectories


def check_plateau_status(trajectory: np.ndarray, window: int = 100) -> bool:
    """Check if trajectory is plateaued using curriculum logic."""
    stats = LessonStats(
        training_stats=PerformanceStats(
            total_samples=len(trajectory), reward_history=trajectory, last_update_step=len(trajectory)
        )
    )
    return is_plateaued(stats, window=window, threshold=0.01)


def find_plateau_point(trajectory: np.ndarray, window: int = 100) -> int | None:
    """Find the first timestep where trajectory becomes plateaued.

    Returns None if never plateaus, otherwise returns the index.
    """
    # Need at least window points to check
    if len(trajectory) < window:
        return None

    for i in range(window, len(trajectory) + 1):
        partial_traj = trajectory[:i]
        stats = LessonStats(
            training_stats=PerformanceStats(
                total_samples=len(partial_traj), reward_history=partial_traj, last_update_step=i
            )
        )
        if is_plateaued(stats, window=window, threshold=0.01):
            return i

    return None


def plot_trajectories():
    """Generate and plot all trajectories with plateau detection."""
    trajectories = generate_trajectories(n_steps=200)

    # Find plateau points for each
    plateau_points = {name: find_plateau_point(traj, window=100) for name, traj in trajectories.items()}

    # Create figure with subplots
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = axes[idx]
        plateau_idx = plateau_points[name]

        # Plot trajectory with color change at plateau point
        if plateau_idx is not None:
            # Before plateau: coral/orange
            ax.plot(range(plateau_idx), traj[:plateau_idx], color="coral", linewidth=1.5, alpha=0.8, label="Pre-plateau")
            # After plateau: green
            ax.plot(
                range(plateau_idx - 1, len(traj)),
                traj[plateau_idx - 1 :],
                color="green",
                linewidth=1.5,
                alpha=0.8,
                label="Plateaued",
            )
            # Mark the transition point
            ax.axvline(
                x=plateau_idx,
                color="darkblue",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Plateau at t={plateau_idx}",
            )
            ax.scatter(
                [plateau_idx],
                [traj[plateau_idx]],
                color="darkblue",
                s=80,
                zorder=5,
                marker="o",
                edgecolors="white",
                linewidth=2,
            )

            title_status = f"PLATEAUED at t={plateau_idx}"
            title_color = "darkgreen"
        else:
            # Never plateaued: all coral
            ax.plot(traj, color="coral", linewidth=1.5, alpha=0.8)
            title_status = "NOT PLATEAUED"
            title_color = "darkred"

        # Style
        ax.set_title(f"{name}\n{title_status}", fontsize=10, fontweight="bold", color=title_color)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Reward", fontsize=9)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        recent = traj[-100:] if len(traj) >= 100 else traj
        mean_recent = np.mean(recent)
        std_recent = np.std(recent)
        stats_text = f"mu={mean_recent:.3f}\n theta={std_recent:.3f}"
        if plateau_idx is not None:
            stats_text += f"\nplateau@{plateau_idx}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()

    # Summary
    n_plateaued = sum(1 for p in plateau_points.values() if p is not None)
    fig.suptitle(
        f"Reward Trajectory Plateau Detection (window=100)\n"
        f"{n_plateaued}/{len(trajectories)} trajectories reached plateau",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    plt.subplots_adjust(top=0.97)
    plt.savefig("plateau_detection_analysis.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: plateau_detection_analysis.png")

    # Print summary
    print("\n" + "=" * 60)
    print("PLATEAU DETECTION SUMMARY")
    print("=" * 60)
    for name, plateau_idx in plateau_points.items():
        if plateau_idx is not None:
            status = f"✓ PLATEAUED at t={plateau_idx}"
        else:
            status = "✗ NOT PLATEAUED"
        print(f"{status:30s} | {name}")
    print("=" * 60)


if __name__ == "__main__":
    plot_trajectories()
