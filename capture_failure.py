"""
StaleMind — Phase 1: Capture LLM Failure Under Drift
Proves that Qwen2.5-7B-Instruct makes wrong decisions when preferences drift.
"""

import requests
import json
import os
from huggingface_hub import InferenceClient

BASE_URL = "https://mrhapile-stalemind.hf.space"
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

VALID_ACTIONS = ["ACCEPT", "REJECT", "ESCALATE", "PROPOSE_RESCHEDULE", "DELEGATE", "DRAFT_MESSAGE"]


def get_reward(result):
    r = result["reward"]
    return r["score"] if isinstance(r, dict) else r


def ask_model(observation):
    """Ask Qwen to decide an action based on the observation."""
    prompt = f"""You are an AI scheduling assistant. You MUST follow the user's stated preferences exactly.

CURRENT STATE:
- Request from boss: {observation['request']}
- Context message: {observation['message']}
- User's stated preferences: {observation['visible_preferences'][0]}
- Boss relationship score: {observation['relationships']['boss']}
- Family relationship score: {observation['relationships']['family']}
- Current step: {observation['step']}

RULES:
- If the user prefers "work > family", you should ACCEPT work requests.
- If the user prefers "family > work", you should REJECT work requests.
- You MUST follow the stated preferences. Do not override them.

Available actions: ACCEPT, REJECT, ESCALATE, PROPOSE_RESCHEDULE, DELEGATE, DRAFT_MESSAGE

Respond with ONLY a JSON object:
{{"type": "<ACTION>", "content": "<brief note>", "reasoning": "<why>"}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat_completion(messages=messages, max_tokens=150)
        raw = resp.choices[0].message.content.strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw)

        if parsed.get("type") not in VALID_ACTIONS:
            parsed["type"] = "ACCEPT"
            parsed["reasoning"] = f"(fallback) original: {raw[:80]}"

        return parsed
    except Exception as e:
        return {"type": "ACCEPT", "content": "parse error", "reasoning": f"Error: {e}"}


def run_episode(scenario_index, scenario_name):
    """Run a full 10-step episode and log every decision."""
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario_name} (index={scenario_index})")
    print(f"{'='*70}")

    r = requests.post(f"{BASE_URL}/reset", json={"scenario_index": scenario_index})
    obs = r.json()["observation"]

    episode_log = {
        "scenario": scenario_name,
        "scenario_index": scenario_index,
        "steps": []
    }

    total_reward = 0.0
    drift_detected = False

    for step in range(10):
        decision = ask_model(obs)
        action_type = decision.get("type", "ACCEPT")
        content = decision.get("content", "")
        reasoning = decision.get("reasoning", "")

        r = requests.post(f"{BASE_URL}/step", json={"type": action_type, "content": content})
        result = r.json()

        reward = get_reward(result)
        done = result["done"]
        new_obs = result["observation"]

        has_drift_signal = any(w in obs.get("message", "").lower()
                              for w in ["son", "event", "home", "needs", "busy"])

        if has_drift_signal and not drift_detected:
            drift_detected = True

        is_failure = has_drift_signal and action_type == "ACCEPT" and reward < 0.5

        step_log = {
            "step": step + 1,
            "observation_message": obs.get("message", ""),
            "action": action_type,
            "content": content,
            "reasoning": reasoning,
            "reward": reward,
            "drift_active": has_drift_signal,
            "is_failure": is_failure,
            "relationships": new_obs.get("relationships", {})
        }
        episode_log["steps"].append(step_log)
        total_reward += reward

        drift_tag = " | DRIFT ACTIVE" if has_drift_signal else ""
        fail_tag = " | *** FAILURE ***" if is_failure else ""
        print(f"  Step {step+1:2d} | {action_type:20s} | reward: {reward:.2f}{drift_tag}{fail_tag}")
        print(f"          Reasoning: {reasoning[:90]}")

        obs = new_obs

        if done:
            break

    episode_log["total_reward"] = total_reward
    print(f"\n  TOTAL REWARD: {total_reward:.2f}")
    return episode_log


if __name__ == "__main__":
    print("=" * 70)
    print("  StaleMind — LLM Failure Capture")
    print("  Model: Qwen/Qwen2.5-7B-Instruct")
    print("  Hypothesis: Model follows stale preferences after hidden drift")
    print("=" * 70)

    all_logs = []
    scenarios = [
        (0, "Easy (drift at step 7)"),
        (1, "Medium (drift at step 5)"),
        (2, "Hard (drift at step 3)")
    ]

    for idx, name in scenarios:
        log = run_episode(idx, name)
        all_logs.append(log)

    with open("qwen_failure_logs.json", "w") as f:
        json.dump(all_logs, f, indent=2)

    print(f"\n{'='*70}")
    print("  RESULTS SAVED: qwen_failure_logs.json")
    print(f"{'='*70}")

    print("\n  SUMMARY:")
    print(f"  {'Scenario':35s} | {'Total':8s} | {'Failures':8s}")
    print(f"  {'-'*60}")
    for log in all_logs:
        failures = sum(1 for s in log["steps"] if s["is_failure"])
        drifted = sum(1 for s in log["steps"] if s["drift_active"])
        print(f"  {log['scenario']:35s} | {log['total_reward']:8.2f} | {failures}/{drifted} steps")
