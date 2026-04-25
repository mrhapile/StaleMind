import json
import re

from env.environment import ACTIONS, StaleMindEnv

VALID_ACTIONS = list(ACTIONS)

FAMILY_WORDS = ["home", "family", "present", "personal", "support", "counting on"]
WORK_WORDS = ["work", "professional", "stakeholder", "deadline", "commitment", "pressure"]
DOUBT_WORDS = ["mixed", "muddy", "inconsistent", "uncertain", "premature"]


def clamp(value, low=0.01, high=0.99):
    return max(low, min(high, value))


def extract_signal_features(message):
    text = message.lower()
    return {
        "family": sum(word in text for word in FAMILY_WORDS),
        "work": sum(word in text for word in WORK_WORDS),
        "doubt": sum(word in text for word in DOUBT_WORDS),
    }


class BayesianBeliefTracker:
    def __init__(self, belief_work=0.80, belief_family=0.20, steps_since_ask=3):
        self.belief_work = clamp(belief_work)
        self.belief_family = clamp(belief_family)
        self.steps_since_ask = steps_since_ask
        self.normalize()

    def normalize(self):
        total = self.belief_work + self.belief_family
        self.belief_work /= total
        self.belief_family /= total

    @property
    def uncertainty(self):
        return 1.0 - abs(self.belief_work - self.belief_family)

    @property
    def dominant(self):
        return "work" if self.belief_work >= self.belief_family else "family"

    def snapshot(self):
        return {
            "belief_work": self.belief_work,
            "belief_family": self.belief_family,
            "steps_since_ask": self.steps_since_ask,
        }

    def update(self, obs, last_action=None, last_reward=0.0):
        features = extract_signal_features(obs["message"])

        update_strength = 1.0
        expected_reward = 0.2
        if last_action is not None and last_reward < expected_reward:
            update_strength *= 1.5

        signal_work = 1.0 + (0.30 * features["work"] + 0.05 * obs["urgency"]) * update_strength
        signal_family = 1.0 + (0.30 * features["family"] + 0.08 * obs["impact"]) * update_strength
        ambiguity = 1.0 + 0.25 * features["doubt"] + 0.12 * (1.0 - obs["reversibility"])

        signal_work *= 1.0 / ambiguity
        signal_family *= 1.0 / ambiguity

        outcome_work = 1.0
        outcome_family = 1.0
        if last_action == "ACCEPT":
            outcome_work += max(0.0, last_reward) * 0.9 * update_strength
            outcome_family += max(0.0, -last_reward) * 1.35 * update_strength
        elif last_action == "REJECT":
            outcome_family += max(0.0, last_reward) * 0.9 * update_strength
            outcome_work += max(0.0, -last_reward) * 1.35 * update_strength
        elif last_action in {"PROPOSE_RESCHEDULE", "DELEGATE"}:
            outcome_family += max(0.0, last_reward) * 0.35 * update_strength
            outcome_work += max(0.0, last_reward) * 0.35 * update_strength
            if last_reward < 0:
                outcome_family += 0.20 * update_strength
                outcome_work += 0.20 * update_strength
        elif last_action == "ASK_CLARIFICATION":
            signal_work += 0.12 * update_strength
            signal_family += 0.12 * update_strength

        prior_work = self.belief_work
        prior_family = self.belief_family
        posterior_work = prior_work * signal_work * outcome_work
        posterior_family = prior_family * signal_family * outcome_family

        self.belief_work = clamp(posterior_work)
        self.belief_family = clamp(posterior_family)
        self.normalize()

        # uncertainty decay
        self.belief_work = 0.5 + (self.belief_work - 0.5) * 0.9
        self.belief_family = 0.5 + (self.belief_family - 0.5) * 0.9

    def choose_action(self, obs):
        if (
            self.uncertainty > 0.52
            and obs["impact"] > 0.55
            and obs["reversibility"] < 0.60
            and self.steps_since_ask > 2
        ):
            self.steps_since_ask = 0
            return (
                "ASK_CLARIFICATION",
                "My belief is unstable and the cost of being wrong is high, so I should ask first.",
            )

        self.steps_since_ask += 1

        if self.belief_family > 0.58:
            if obs["commitment_load"] > 0.75 and obs["reversibility"] > 0.45:
                return (
                    "PROPOSE_RESCHEDULE",
                    "Family currently looks more likely, but there is still room to recover with a reversible move.",
                )
            if obs["delegation_feasibility"] > 0.72 and obs["urgency"] > 0.68:
                return (
                    "DELEGATE",
                    "Family looks more likely, but urgency is real, so delegation reduces downside.",
                )
            return (
                "REJECT",
                "Recent evidence and outcomes make family the more likely priority.",
            )

        if self.belief_work > 0.64:
            if obs["commitment_load"] > 1.05 and obs["delegation_feasibility"] > 0.70:
                return (
                    "DELEGATE",
                    "Work still looks more likely, but commitment load is high enough to justify delegation.",
                )
            return (
                "ACCEPT",
                "Work remains the most likely priority given recent evidence and outcomes.",
            )

        if obs["reversibility"] > 0.60:
            return (
                "PROPOSE_RESCHEDULE",
                "I am uncertain, so I should preserve optionality instead of hard committing.",
            )

        return (
            "ESCALATE",
            "The evidence is unresolved and the decision is hard to reverse, so escalation is safer.",
        )


def format_obs_as_prompt(obs, step_num, tracker=None):
    visible_prefs = obs.get("visible_preferences", ["work > family"])
    if isinstance(visible_prefs, list):
        visible_prefs = ", ".join(visible_prefs)

    belief_line = ""
    if tracker is not None:
        belief_line = (
            f"Current internal belief: work={tracker.belief_work:.2f}, "
            f"family={tracker.belief_family:.2f}, uncertainty={tracker.uncertainty:.2f}\n"
        )

    return (
        "You are an AI assistant managing a user's schedule under uncertainty.\n"
        f"Visible preference memory: {visible_prefs}\n"
        f"Step {step_num}/10\n"
        f"Message: {obs.get('message', '')}\n"
        f"Urgency={obs.get('urgency', 0.0):.2f}, Impact={obs.get('impact', 0.0):.2f}, "
        f"Reversibility={obs.get('reversibility', 0.0):.2f}, Delegation={obs.get('delegation_feasibility', 0.0):.2f}\n"
        f"Commitment load={obs.get('commitment_load', 0.0):.2f}\n"
        f"{belief_line}"
        f"Choose exactly one action from: {', '.join(VALID_ACTIONS)}\n"
        'Respond with JSON: {"action": "<ACTION>", "reasoning": "<why>"}'
    )


def parse_action_from_completion(text):
    text = text.strip()
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            action = parsed.get("action", parsed.get("type", "")).upper().strip()
            if action in VALID_ACTIONS:
                return action, True
    except Exception:
        pass

    text_upper = text.upper()
    for action in VALID_ACTIONS:
        if action in text_upper:
            return action, False
    return "ESCALATE", False


def rollout_policy(
    scenario_index=1,
    seed=0,
    config=None,
    policy_kind="adaptive",
    forced_first_action=None,
    initial_tracker=None,
):
    env = StaleMindEnv(seed=seed)
    obs, _ = env.reset(scenario_index=scenario_index, config=config)
    tracker = initial_tracker or BayesianBeliefTracker()

    total_reward = 0.0
    belief_errors = []
    detection_steps = {}
    drift_events = env.debug_state()["drift_events"]
    last_action = None
    last_reward = 0.0
    steps = []

    for step_num in range(1, env.state_dict["max_steps"] + 1):
        tracker.update(obs, last_action, last_reward)

        if forced_first_action is not None and step_num == 1:
            action = forced_first_action
            reasoning = "Forced first action for evaluation."
        elif policy_kind == "adaptive":
            action, reasoning = tracker.choose_action(obs)
        elif policy_kind == "always_accept":
            action, reasoning = "ACCEPT", "Static accept policy."
        elif policy_kind == "step_threshold":
            action = "REJECT" if obs["step"] >= 5 else "ACCEPT"
            reasoning = "Static threshold policy."
        elif policy_kind == "keyword":
            features = extract_signal_features(obs["message"])
            action = "REJECT" if features["family"] > features["work"] else "ACCEPT"
            reasoning = "Keyword-driven policy."
        elif policy_kind == "always_ask":
            action, reasoning = "ASK_CLARIFICATION", "Always ask."
        elif policy_kind == "random":
            action = env.rng.choice(VALID_ACTIONS)
            reasoning = "Random policy."
        else:
            raise ValueError(f"Unknown policy kind: {policy_kind}")

        new_obs, reward, done, info = env.step({"type": action, "content": ""})
        
        # Punish lucky correctness
        action_correct = reward > 0.0
        belief_confidence = abs(tracker.belief_work - tracker.belief_family)
        if action_correct and belief_confidence < 0.5:
            reward -= 0.2

        debug = env.debug_state()
        true_work = debug["true_weights"]["work"]
        belief_error = abs(tracker.belief_work - true_work)
        belief_errors.append(belief_error)

        for event in drift_events:
            if step_num < event["step"] or event["step"] in detection_steps:
                continue
            if tracker.dominant == event["target"] and abs(tracker.belief_work - tracker.belief_family) > 0.12:
                detection_steps[event["step"]] = step_num

        steps.append(
            {
                "step": step_num,
                "action": action,
                "reasoning": reasoning,
                "reward": reward,
                "belief_work": tracker.belief_work,
                "truth_work": true_work,
                "belief_error": belief_error,
                "info": info,
            }
        )

        total_reward += reward
        obs = new_obs
        last_action = action
        last_reward = reward
        if done:
            break

    latencies = []
    recovery_qualities = []
    for event in drift_events:
        detected = detection_steps.get(event["step"])
        if detected is None:
            latencies.append(env.state_dict["max_steps"] - event["step"] + 1)
            recovery_qualities.append(0.0)
            continue
        latency = max(0, detected - event["step"])
        latencies.append(latency)
        recovery_qualities.append(max(0.0, 1.0 - latency * 0.2))

    mean_belief_error = (
        sum(belief_errors) / len(belief_errors) if belief_errors else 1.0
    )
    belief_accuracy = 1.0 - mean_belief_error
    
    recoveries = 0
    errors = 0
    for i in range(1, len(steps)):
        if steps[i-1]["reward"] < 0:
            errors += 1
            if steps[i]["reward"] > 0:
                recoveries += 1
    recovery_quality = (recoveries / errors) if errors > 0 else 1.0

    actions = [s["action"] for s in steps]
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
    decision_consistency = max(0.0, 1.0 - switches * 0.1)

    adaptation_score = (
        0.5 * belief_accuracy +
        0.3 * recovery_quality +
        0.2 * decision_consistency
    )

    adapt_bonus = 1.0 - mean_belief_error
    latency_penalty = sum(latencies) * 0.2
    final_reward = total_reward + adapt_bonus - latency_penalty

    return {
        "env_reward": total_reward,
        "cumulative_reward": total_reward,
        "belief_error": mean_belief_error,
        "adapt_bonus": adapt_bonus,
        "latency_penalty": latency_penalty,
        "final_reward": final_reward,
        "adaptation_score": adaptation_score,
        "drift_events": drift_events,
        "detection_steps": detection_steps,
        "steps": steps,
    }


def build_training_samples(num_scenarios=3, episodes_per_scenario=12):
    samples = []
    for scenario_index in range(num_scenarios):
        for offset in range(episodes_per_scenario):
            seed = scenario_index * 1000 + offset
            for config in [
                {"false_signal": False, "delay_drift": False, "force_conflict": False},
                {"false_signal": True, "delay_drift": False, "force_conflict": False},
                {"false_signal": False, "delay_drift": True, "force_conflict": True},
            ]:
                env = StaleMindEnv(seed=seed)
                obs, _ = env.reset(scenario_index=scenario_index, config=config)
                tracker = BayesianBeliefTracker()
                last_action = None
                last_reward = 0.0
                action_history = []

                for step_num in range(1, env.state_dict["max_steps"] + 1):
                    tracker.update(obs, last_action, last_reward)
                    samples.append(
                        {
                            "prompt": format_obs_as_prompt(obs, step_num, tracker),
                            "scenario_index": scenario_index,
                            "seed": seed,
                            "config_json": json.dumps(config, sort_keys=True),
                        }
                    )

                    rollout_action, _ = tracker.choose_action(obs)
                    obs, reward, done, _ = env.step(
                        {"type": rollout_action, "content": ""}
                    )
                    action_history.append(rollout_action)
                    last_action = rollout_action
                    last_reward = reward
                    if done:
                        break
    return samples


def evaluate_completion_reward(prompt, completion, scenario_index, seed, config_json):
    del prompt
    action, valid = parse_action_from_completion(completion)
    config = json.loads(config_json)
    rollout = rollout_policy(
        scenario_index=int(scenario_index),
        seed=int(seed),
        config=config,
        policy_kind="adaptive",
        forced_first_action=action,
        initial_tracker=BayesianBeliefTracker(),
    )
    if not valid:
        rollout["final_reward"] -= 0.15
    return rollout
