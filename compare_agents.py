"""
StaleMind — Phase 3: Naive vs Adaptive Agent Comparison
Generates grouped bar chart showing adaptive agent outperforms naive agent.
"""

import requests
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "https://mrhapile-stalemind.hf.space"


def get_reward(result):
    r = result["reward"]
    return r["score"] if isinstance(r, dict) else r


def naive_agent(obs):
    """Always accepts — ignores all signals."""
    return {"type": "ACCEPT", "content": ""}


def adaptive_agent(obs):
    """Detects drift signals and adjusts behavior."""
    msg = obs.get("message", "").lower()

    if any(w in msg for w in ["son", "event", "home", "needs you", "busy", "family"]):
        return {"type": "REJECT", "content": "family priority"}

    return {"type": "ACCEPT", "content": "work priority"}


def run_episode(agent_fn, scenario_index):
    """Run a full episode with a given agent function."""
    r = requests.post(f"{BASE_URL}/reset", json={"scenario_index": scenario_index})
    obs = r.json()["observation"]

    total_reward = 0.0

    for _ in range(10):
        action = agent_fn(obs)
        r = requests.post(f"{BASE_URL}/step", json=action)
        result = r.json()

        reward = get_reward(result)
        total_reward += reward
        obs = result["observation"]

        if result["done"]:
            break

    return total_reward


def main():
    scenarios = ["Easy", "Medium", "Hard"]

    print("Running Naive Agent...")
    naive_scores = []
    for i in range(3):
        score = run_episode(naive_agent, i)
        naive_scores.append(score)
        print(f"  {scenarios[i]:8s}: {score:.2f}")

    print("\nRunning Adaptive Agent...")
    adaptive_scores = []
    for i in range(3):
        score = run_episode(adaptive_agent, i)
        adaptive_scores.append(score)
        print(f"  {scenarios[i]:8s}: {score:.2f}")

    # Save data
    comparison_data = {
        "scenarios": scenarios,
        "naive_scores": naive_scores,
        "adaptive_scores": adaptive_scores
    }
    with open("comparison_data.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # ---- Generate Grouped Bar Chart ----
    x = np.arange(len(scenarios))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    naive_color = "#e74c3c"
    adaptive_color = "#2ecc71"

    bars1 = ax.bar(x - width/2, naive_scores, width, label="Naive Agent (always ACCEPT)",
                   color=naive_color, edgecolor="white", linewidth=1.2, zorder=3)
    bars2 = ax.bar(x + width/2, adaptive_scores, width, label="Adaptive Agent (drift-aware)",
                   color=adaptive_color, edgecolor="white", linewidth=1.2, zorder=3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontweight="bold",
                fontsize=11, color=naive_color)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontweight="bold",
                fontsize=11, color=adaptive_color)

    ax.set_xlabel("Scenario Difficulty", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Episode Reward", fontsize=13, fontweight="bold")
    ax.set_title("StaleMind: Naive vs Adaptive Agent Under Preference Drift",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, max(max(naive_scores), max(adaptive_scores)) + 2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nChart saved: comparison.png")

    # Print summary
    print(f"\n{'='*50}")
    print(f"{'Scenario':10s} | {'Naive':8s} | {'Adaptive':8s} | {'Delta':8s}")
    print(f"{'-'*50}")
    for i, s in enumerate(scenarios):
        delta = adaptive_scores[i] - naive_scores[i]
        print(f"{s:10s} | {naive_scores[i]:8.2f} | {adaptive_scores[i]:8.2f} | {delta:+8.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
