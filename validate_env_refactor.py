import json

from stalemind_learning import build_training_samples, rollout_policy

CONFIG = {
    "false_signal": True,
    "delay_drift": True,
    "force_conflict": True,
}


def average_result(policy_kind, seeds=30):
    rewards = []
    scores = []
    for seed in range(seeds):
        rollout = rollout_policy(
            scenario_index=1,
            seed=seed,
            config=CONFIG,
            policy_kind=policy_kind,
        )
        rewards.append(rollout["final_reward"])
        scores.append(rollout["adaptation_score"])
    return sum(rewards) / len(rewards), sum(scores) / len(scores)


def print_policy_suite():
    print("FINAL POLICY SUITE")
    for policy in [
        "always_accept",
        "step_threshold",
        "keyword",
        "always_ask",
        "adaptive",
        "random",
    ]:
        avg_reward, avg_score = average_result(policy)
        print(
            f"{policy:15s} reward={avg_reward:+.3f} adaptation_score={avg_score:.3f}"
        )
    print()


def print_belief_trace():
    rollout = rollout_policy(
        scenario_index=1,
        seed=5,
        config=CONFIG,
        policy_kind="adaptive",
    )
    print("BELIEF TRACE")
    print("drift_events:", rollout["drift_events"])
    print("detection_steps:", rollout["detection_steps"])
    print(
        "belief_work:",
        [round(step["belief_work"], 3) for step in rollout["steps"]],
    )
    print(
        "truth_work:",
        [round(step["truth_work"], 3) for step in rollout["steps"]],
    )
    print(
        "belief_error:",
        [round(step["belief_error"], 3) for step in rollout["steps"]],
    )
    print()


def print_fast_vs_slow():
    fast = rollout_policy(
        scenario_index=1,
        seed=5,
        config=CONFIG,
        policy_kind="adaptive",
    )
    slow = rollout_policy(
        scenario_index=1,
        seed=5,
        config=CONFIG,
        policy_kind="always_accept",
    )
    print("FAST VS SLOW ADAPTATION")
    print(
        "fast:",
        round(fast["adaptation_score"], 3),
        round(fast["final_reward"], 3),
        fast["detection_steps"],
    )
    print(
        "slow:",
        round(slow["adaptation_score"], 3),
        round(slow["final_reward"], 3),
        slow["detection_steps"],
    )
    print()


def print_training_sample_check():
    samples = build_training_samples(num_scenarios=1, episodes_per_scenario=2)
    print("TRAINING SAMPLE CHECK")
    print("sample_count:", len(samples))
    print("sample_keys:", sorted(samples[0].keys()))
    print("config_json:", samples[0]["config_json"])
    print("prompt_head:", samples[0]["prompt"][:220])
    print("config_decoded:", json.loads(samples[0]["config_json"]))
    print()


if __name__ == "__main__":
    print_policy_suite()
    print_belief_trace()
    print_fast_vs_slow()
    print_training_sample_check()
