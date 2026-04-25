"""
StaleMind — Phase 4: Gradio Interactive Demo
Lets judges see drift happening, wrong decisions, and reward drops in real-time.
"""

import gradio as gr
import requests
import uuid

BASE_URL = "http://localhost:8000"

SCENARIOS = {
    "Easy (drift at step 7)": 0,
    "Medium (drift at step 5)": 1,
    "Hard (drift at step 3)": 2
}


def get_reward(result):
    r = result["reward"]
    return r["score"] if isinstance(r, dict) else r


def run_full_episode(scenario_name, agent_type):
    """Run a full episode and return formatted results."""
    scenario_index = SCENARIOS[scenario_name]

    session_id = str(uuid.uuid4())
    
    # Reset
    r = requests.post(f"{BASE_URL}/reset", json={"scenario_index": scenario_index, "session_id": session_id})
    obs = r.json()["observation"]

    log_lines = []
    total_reward = 0.0
    failure_count = 0

    log_lines.append(f"{'='*65}")
    log_lines.append(f"  Scenario: {scenario_name}")
    log_lines.append(f"  Agent: {agent_type}")
    log_lines.append(f"{'='*65}\n")

    for step in range(10):
        msg = obs.get("message", "").lower()

        # Choose action based on agent type
        if agent_type == "Naive (always ACCEPT)":
            action = {"type": "ACCEPT", "content": ""}
            reasoning = "Always accept work requests regardless of context"
        else:
            if any(w in msg for w in ["son", "event", "home", "needs you", "busy", "family"]):
                action = {"type": "REJECT", "content": "family priority"}
                reasoning = "Detected family signal in message -> rejecting work request"
            else:
                action = {"type": "ACCEPT", "content": "work priority"}
                reasoning = "No drift signal detected -> following work preference"

        # Take step
        action["session_id"] = session_id
        r = requests.post(f"{BASE_URL}/step", json=action)
        result = r.json()

        reward = get_reward(result)
        done = result["done"]
        new_obs = result["observation"]

        has_drift = any(w in msg for w in ["son", "event", "home", "needs", "busy"])
        is_failure = has_drift and action["type"] == "ACCEPT" and reward < 0.5
        if is_failure:
            failure_count += 1

        total_reward += reward

        # Format output
        drift_tag = "  [DRIFT]" if has_drift else ""
        fail_tag = "  *** FAILURE ***" if is_failure else ""
        reward_bar = "+" * int(reward * 10) + "-" * (10 - int(reward * 10))

        log_lines.append(f"  Step {step+1:2d} | {action['type']:12s} | [{reward_bar}] {reward:.2f}{drift_tag}{fail_tag}")
        log_lines.append(f"           {reasoning}")
        log_lines.append(f"           Boss: {new_obs['relationships']['boss']:.2f}  Family: {new_obs['relationships']['family']:.2f}")
        log_lines.append("")

        obs = new_obs
        if done:
            break

    log_lines.append(f"{'='*65}")
    log_lines.append(f"  TOTAL REWARD: {total_reward:.2f}")
    log_lines.append(f"  FAILURES: {failure_count}")
    log_lines.append(f"{'='*65}")

    return "\n".join(log_lines), f"{total_reward:.2f}", str(failure_count)


def run_comparison(scenario_name):
    """Run both agents and compare."""
    naive_log, naive_total, naive_fails = run_full_episode(scenario_name, "Naive (always ACCEPT)")
    adaptive_log, adaptive_total, adaptive_fails = run_full_episode(scenario_name, "Adaptive (drift-aware)")

    comparison = f"""
{'='*65}
  COMPARISON SUMMARY
{'='*65}

  Scenario: {scenario_name}

  Agent              | Total Reward | Failures
  --------------------|-------------|----------
  Naive (ACCEPT all)  | {naive_total:>11s} | {naive_fails}
  Adaptive (aware)    | {adaptive_total:>11s} | {adaptive_fails}

  Delta: {float(adaptive_total) - float(naive_total):+.2f} reward improvement

{'='*65}
"""

    return naive_log, adaptive_log, comparison


# Build Gradio UI
with gr.Blocks(
    title="StaleMind — Drift Detection Demo"
) as demo:

    gr.Markdown("""
    # StaleMind — Preference Drift Simulation

    An RL-style environment demonstrating how AI agents fail when their understanding becomes **stale**.

    The environment presents work vs family scheduling conflicts. Mid-episode, the user's **true preferences shift**
    from "work > family" to "family > work" — but the agent only sees the original, stale preferences.

    ---
    """)

    with gr.Tab("Single Agent Run"):
        with gr.Row():
            scenario_input = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value="Medium (drift at step 5)",
                label="Scenario"
            )
            agent_input = gr.Dropdown(
                choices=["Naive (always ACCEPT)", "Adaptive (drift-aware)"],
                value="Naive (always ACCEPT)",
                label="Agent Type"
            )
            run_btn = gr.Button("Run Episode", variant="primary")

        with gr.Row():
            total_output = gr.Textbox(label="Total Reward", interactive=False)
            fail_output = gr.Textbox(label="Failures", interactive=False)

        log_output = gr.Textbox(label="Episode Log", lines=30, interactive=False)

        run_btn.click(
            fn=run_full_episode,
            inputs=[scenario_input, agent_input],
            outputs=[log_output, total_output, fail_output]
        )

    with gr.Tab("Naive vs Adaptive Comparison"):
        comp_scenario = gr.Dropdown(
            choices=list(SCENARIOS.keys()),
            value="Medium (drift at step 5)",
            label="Scenario"
        )
        comp_btn = gr.Button("Run Comparison", variant="primary")

        with gr.Row():
            naive_output = gr.Textbox(label="Naive Agent Log", lines=25, interactive=False)
            adaptive_output = gr.Textbox(label="Adaptive Agent Log", lines=25, interactive=False)

        comp_summary = gr.Textbox(label="Comparison Summary", lines=12, interactive=False)

        comp_btn.click(
            fn=run_comparison,
            inputs=[comp_scenario],
            outputs=[naive_output, adaptive_output, comp_summary]
        )

    gr.Markdown("""
    ---
    **How it works:**
    - The environment runs 10 steps per episode
    - At a specific step (varies by difficulty), the hidden `true_preferences` flip
    - The agent only sees `visible_preferences: ["work > family"]` (never updated)
    - Soft signals appear in the observation message (e.g., "Your son needs you")
    - The **Naive agent** ignores these signals and keeps ACCEPTing → reward drops to ~0
    - The **Adaptive agent** detects drift signals and switches to REJECT → maintains high reward
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
