"""
StaleMind — belief-update demo.
Shows confidence, doubt, adaptation, and recovery rather than a plain action log.
"""

import uuid

from stalemind_learning import rollout_policy

try:
    from fastapi.middleware.cors import CORSMiddleware
    from main import app as api
except ImportError:  # pragma: no cover - allows local validation without web deps installed
    CORSMiddleware = None
    api = None

try:
    import gradio as gr
except ImportError:  # pragma: no cover - allows local validation without gradio installed
    gr = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - allows local validation without matplotlib installed
    matplotlib = None
    plt = None

if api is not None and CORSMiddleware is not None:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

SCENARIOS = {
    "Easy (cleaner evidence)": 0,
    "Medium (ambiguous evidence)": 1,
    "Hard (delayed + conflicting evidence)": 2,
}

AGENTS = [
    "Step Threshold (heuristic)",
    "Belief-Updating Agent",
]


def render_plot(episode):
    if plt is None:
        return None
    steps = episode["steps"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(steps, episode["belief_work"], color="#1f77b4", linewidth=2.5, label="Belief: work")
    axes[0].plot(steps, episode["truth_work"], color="#111111", linewidth=2.0, linestyle="--", label="Reality: work")
    axes[0].plot(steps, episode["belief_family"], color="#d62728", linewidth=2.0, alpha=0.7, label="Belief: family")
    axes[0].set_ylabel("Belief")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Belief Compass")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, episode["belief_error"], color="#9467bd", linewidth=2.2)
    axes[1].set_ylabel("Error")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Belief Error")
    axes[1].grid(alpha=0.25)

    reward_colors = ["#2ca02c" if reward >= 0 else "#ff7f0e" for reward in episode["rewards"]]
    axes[2].bar(steps, episode["rewards"], color=reward_colors, alpha=0.85)
    axes[2].plot(steps, episode["commitment_loads"], color="#444444", linewidth=1.8, label="Commitment load")
    axes[2].axhline(0, color="#888888", linewidth=1.0)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Reward / Load")
    axes[2].set_title("Timeline")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    return fig


def run_episode_data(
    scenario_name,
    agent_type,
    inject_false_signal=False,
    delay_drift=False,
    force_conflict=False,
):
    run_config = {
        "false_signal": inject_false_signal,
        "delay_drift": delay_drift,
        "force_conflict": force_conflict,
    }
    scenario_index = SCENARIOS[scenario_name]
    if agent_type == "Step Threshold (heuristic)":
        policy_kind = "step_threshold"
    else:
        policy_kind = "adaptive"

    rollout = rollout_policy(
        scenario_index=scenario_index,
        seed=uuid.uuid4().int % 10_000_000,
        config=run_config,
        policy_kind=policy_kind,
    )

    belief_lines = []
    thought_lines = []
    timeline_lines = []

    episode = {
        "steps": [],
        "belief_work": [],
        "belief_family": [],
        "truth_work": [],
        "belief_error": [],
        "rewards": [],
        "commitment_loads": [],
    }
    failure_count = 0
    recovery_count = 0
    last_reward = 0.0

    for entry in rollout["steps"]:
        reward = entry["reward"]
        belief_work = entry["belief_work"]
        belief_family = 1.0 - belief_work
        belief_error = entry["belief_error"]
        components = entry["info"]["reward_components"]

        if reward < 0 and entry["action"] not in {"ASK_CLARIFICATION", "ESCALATE"}:
            failure_count += 1
        if last_reward < 0 and reward > 0.45:
            recovery_count += 1

        belief_lines.append(
            f"[Step {entry['step']}] Belief work {belief_work:.0%} | "
            f"Reality work {entry['truth_work']:.0%} | Error {belief_error:.0%}"
        )

        uncertainty = 1.0 - abs(belief_work - belief_family)
        if belief_error > 0.30:
            doubt_reason = "Recent outcomes and cues are contradicting my prior, so I am updating."
        elif uncertainty > 0.40:
            doubt_reason = "The evidence is mixed, so I am carrying uncertainty forward."
        else:
            dominant = "work" if belief_work >= belief_family else "family"
            doubt_reason = f"My posterior currently favors {dominant}."
        thought_lines.append(f"[Step {entry['step']}] {doubt_reason}")
        thought_lines.append(f"Action: {entry['action']} | Why: {entry['reasoning']}")

        tags = []
        if entry["info"].get("drift_events"):
            tags.append("DRIFT")
        if uncertainty > 0.40:
            tags.append("DOUBT")
        if belief_error > 0.30:
            tags.append("BELIEF UPDATE")
        if reward < 0:
            tags.append("FAILURE")
        if last_reward < 0 and reward > 0.45:
            tags.append("RECOVERY")
        tag_text = f" [{' | '.join(tags)}]" if tags else ""

        timeline_lines.append(
            f"Step {entry['step']:02d} | {entry['action']:18s} | reward {reward:+.2f} | "
            f"belief error {belief_error:.2f}{tag_text}"
        )
        timeline_lines.append(
            f"           reward components: alignment={components.get('alignment', 0):+.2f}, "
            f"commitment={components.get('commitment_pressure', 0):+.2f}, "
            f"repeat={components.get('repetition_penalty', 0):+.2f}, "
            f"ask={components.get('ask_penalty', 0):+.2f}, delayed={components.get('delayed_penalty', 0):+.2f}"
        )

        episode["steps"].append(entry["step"])
        episode["belief_work"].append(belief_work)
        episode["belief_family"].append(belief_family)
        episode["truth_work"].append(entry["truth_work"])
        episode["belief_error"].append(belief_error)
        episode["rewards"].append(reward)
        episode["commitment_loads"].append(
            components.get("commitment_pressure", 0.0) * -1.0
        )
        last_reward = reward

    total_reward = rollout["cumulative_reward"]
    adaptation_score = rollout["adaptation_score"]
    summary = (
        f"Scenario: {scenario_name}\n"
        f"Agent: {agent_type}\n"
        f"Total reward: {total_reward:.2f}\n"
        f"Negative decisions: {failure_count}\n"
        f"Recoveries: {recovery_count}\n"
        f"Adaptation score: {adaptation_score:.2f}\n"
        f"Final belief: work {episode['belief_work'][-1]:.0%} / family {episode['belief_family'][-1]:.0%}"
    )

    return {
        "belief_text": "\n".join(belief_lines),
        "thought_text": "\n".join(thought_lines),
        "timeline_text": "\n".join(timeline_lines),
        "summary": summary,
        "plot": render_plot(episode),
        "total_reward": total_reward,
        "failures": failure_count,
        "adaptation_score": adaptation_score,
    }


def run_single_episode(
    scenario_name,
    agent_type,
    inject_false_signal,
    delay_drift,
    force_conflict,
):
    episode = run_episode_data(
        scenario_name,
        agent_type,
        inject_false_signal,
        delay_drift,
        force_conflict,
    )
    return (
        episode["belief_text"],
        episode["thought_text"],
        episode["timeline_text"],
        episode["summary"],
        episode["plot"],
    )


def run_comparison(
    scenario_name,
    inject_false_signal,
    delay_drift,
    force_conflict,
):
    baseline = run_episode_data(
        scenario_name,
        "Step Threshold (heuristic)",
        inject_false_signal,
        delay_drift,
        force_conflict,
    )
    adaptive = run_episode_data(
        scenario_name,
        "Belief-Updating Agent",
        inject_false_signal,
        delay_drift,
        force_conflict,
    )
    summary = (
        f"Scenario: {scenario_name}\n\n"
        f"Policy A (Step Threshold) reward: {baseline['total_reward']:.2f} | adaptation score: {baseline['adaptation_score']:.2f}\n"
        f"Policy B (Adaptive) reward: {adaptive['total_reward']:.2f} | adaptation score: {adaptive['adaptation_score']:.2f}\n"
        f"Reward delta: {adaptive['total_reward'] - baseline['total_reward']:+.2f}"
    )
    return baseline["plot"], adaptive["plot"], summary


if gr is not None:
    with gr.Blocks(title="StaleMind — Belief Correction Demo") as demo:
        gr.Markdown(
            """
            # StaleMind — Belief Correction Under Drift

            This demo is built around one question:
            **Can an agent notice that its understanding of the user has become stale, update its belief, and recover?**
            """
        )

        with gr.Tab("Belief Run"):
            with gr.Row():
                scenario_input = gr.Dropdown(
                    choices=list(SCENARIOS.keys()),
                    value="Medium (ambiguous evidence)",
                    label="Scenario",
                )
                agent_input = gr.Dropdown(
                    choices=AGENTS,
                    value="Belief-Updating Agent",
                    label="Agent",
                )
                run_btn = gr.Button("Run Episode", variant="primary")

            with gr.Row():
                false_signal_input = gr.Checkbox(label="Inject False Signal")
                delay_drift_input = gr.Checkbox(label="Delay Drift")
                force_conflict_input = gr.Checkbox(label="Force Conflict")

            summary_output = gr.Textbox(label="Summary", lines=6, interactive=False)
            belief_plot = gr.Plot(label="Belief + Timeline")

            with gr.Row():
                belief_output = gr.Textbox(label="Belief Compass", lines=16, interactive=False)
                thought_output = gr.Textbox(label="Agent Thought Stream", lines=16, interactive=False)
                timeline_output = gr.Textbox(label="Timeline", lines=16, interactive=False)

            run_btn.click(
                fn=run_single_episode,
                inputs=[
                    scenario_input,
                    agent_input,
                    false_signal_input,
                    delay_drift_input,
                    force_conflict_input,
                ],
                outputs=[belief_output, thought_output, timeline_output, summary_output, belief_plot],
            )

        with gr.Tab("Baseline vs Belief Agent"):
            compare_scenario = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value="Medium (ambiguous evidence)",
                label="Scenario",
            )
            compare_btn = gr.Button("Run Comparison", variant="primary")

            with gr.Row():
                compare_false_signal = gr.Checkbox(label="Inject False Signal")
                compare_delay_drift = gr.Checkbox(label="Delay Drift")
                compare_force_conflict = gr.Checkbox(label="Force Conflict")

            with gr.Row():
                baseline_plot = gr.Plot(label="Policy A: Step Threshold")
                adaptive_plot = gr.Plot(label="Policy B: Adaptive (belief-based)")

            comparison_summary = gr.Textbox(label="Comparison Summary", lines=6, interactive=False)

            compare_btn.click(
                fn=run_comparison,
                inputs=[
                    compare_scenario,
                    compare_false_signal,
                    compare_delay_drift,
                    compare_force_conflict,
                ],
                outputs=[baseline_plot, adaptive_plot, comparison_summary],
            )

        gr.Markdown(
            """
            ---
            **Demo structure**

            - **Belief Compass**: how the agent's internal confidence moves between work and family
            - **Thought Stream**: where doubt, belief revision, and recovery become visible
            - **Timeline**: action -> reward -> belief error, so judges can see cause and effect
            """
        )

    app = gr.mount_gradio_app(api, demo, path="/") if api is not None else None
else:
    app = api

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
