import json
import random

ACTIONS = [
    "ACCEPT",
    "REJECT",
    "ESCALATE",
    "PROPOSE_RESCHEDULE",
    "DELEGATE",
    "DRAFT_MESSAGE",
    "ASK_CLARIFICATION",
]

SCENARIO_LEVELS = {
    0: "easy",
    1: "medium",
    2: "hard",
}

SCENARIO_PROFILES = {
    "easy": {
        "first_drift_range": (3, 4),
        "second_gap_range": (2, 3),
        "signal_noise": 0.20,
        "missing_signal_rate": 0.20,
        "false_signal_rate": 0.08,
        "conflict_signal_rate": 0.10,
    },
    "medium": {
        "first_drift_range": (3, 5),
        "second_gap_range": (2, 3),
        "signal_noise": 0.28,
        "missing_signal_rate": 0.28,
        "false_signal_rate": 0.12,
        "conflict_signal_rate": 0.14,
    },
    "hard": {
        "first_drift_range": (2, 4),
        "second_gap_range": (2, 3),
        "signal_noise": 0.36,
        "missing_signal_rate": 0.34,
        "false_signal_rate": 0.16,
        "conflict_signal_rate": 0.18,
    },
}

REQUEST_LIBRARY = [
    "A client review landed in the same block as family plans.",
    "A late sync overlaps with time already promised at home.",
    "A deadline meeting was moved into the evening schedule.",
    "A planning review now collides with personal obligations.",
    "A senior stakeholder wants immediate support during family hours.",
]

NEUTRAL_OPENERS = [
    "Two obligations are pulling on the same block of time.",
    "The calendar has tightened around one difficult evening decision.",
    "The schedule is compressed and both sides feel important right now.",
    "A work request and a personal obligation are competing for attention.",
]

FAMILY_CUES = [
    "Someone at home is quietly counting on you to be present tonight.",
    "A family commitment is starting to matter more than it first appeared.",
    "Support at home may be more important than the calendar first suggested.",
    "The personal side of this conflict is becoming harder to dismiss.",
]

WORK_CUES = [
    "The work side of this request could create downstream pressure if delayed.",
    "The professional cost of slipping this request looks higher than usual.",
    "There is extra pressure to keep the work commitment intact tonight.",
    "The work request is starting to feel more consequential than it first seemed.",
]

CONFLICT_CUES = [
    "The clues are mixed and neither side is clearly dominant yet.",
    "The context is muddy, and a quick reaction could be premature.",
    "The evidence is inconsistent, so confidence should probably drop.",
]

COMMITMENT_NOTES = [
    "Earlier accepted work is still consuming time in the background.",
    "A previous commitment is still taking attention away from home.",
    "Past decisions are narrowing how flexible the next move can be.",
]

MAX_HISTORY = 100


class StaleMindEnv:
    def __init__(self, seed=None, history_length=MAX_HISTORY):
        self.rng = random.Random(seed)
        self.history_length = history_length
        self.history = []
        self.action_window = []
        self.episode_plan = None
        self.run_config = {}
        self.state_dict = {}

    def seed(self, seed):
        self.rng.seed(seed)

    def reset(self, scenario_index=None, config=None):
        self.run_config = config or {}
        self.episode_plan = self._generate_episode(scenario_index, self.run_config)
        initial_context = self.episode_plan["contexts"][0]

        self.history = []
        self.action_window = []
        self.state_dict = {
            "current_step": 0,
            "max_steps": self.episode_plan["max_steps"],
            "done": False,
            "preferences": ["work > family"],
            "relationships": {"boss": 0.62, "family": 0.62},
            "true_weights": dict(self.episode_plan["initial_true_weights"]),
            "pending_commitments": [],
            "pending_penalties": [],
            "ask_count": 0,
            "clarification_bonus": 0,
            "applied_drift_steps": [],
            "last_action": None,
            "last_info": {},
            "current_context": initial_context,
            "current_request": initial_context["request"],
            "current_message": "",
        }
        self.state_dict["current_message"] = self._build_message()
        return self.state(), {}

    def step(self, action):
        if self.state_dict["done"]:
            return None, 0.0, True, {}

        self.state_dict["current_step"] += 1
        drift_effect = self._apply_drift(self.state_dict["current_step"])
        delayed_effect = self._apply_pending_penalties(self.state_dict["current_step"])
        commitment_effect = self._advance_commitments()

        action_type, content, error = self._parse_action(action)
        if error is not None:
            reward = max(
                -1.0,
                min(
                    1.5,
                    -0.4
                    + delayed_effect["reward_penalty"]
                    + commitment_effect["reward_penalty"],
                ),
            )
            action_type = "INVALID"
            reward_components = {
                "format_penalty": -0.4,
                "delayed_penalty": round(delayed_effect["reward_penalty"], 3),
                "commitment_pressure": round(commitment_effect["reward_penalty"], 3),
            }
        else:
            if action_type == "ASK_CLARIFICATION":
                self.state_dict["ask_count"] += 1

            reward, reward_components = self._compute_reward(
                action_type,
                content,
                commitment_effect["reward_penalty"],
                delayed_effect["reward_penalty"],
            )
            self._apply_action_effects(action_type)
            self._apply_action_relationships(action_type)
            self._schedule_delayed_consequence(action_type)
            self._apply_repetition_tracking(action_type)
            self.state_dict["last_action"] = action_type

            if action_type == "ASK_CLARIFICATION":
                self.state_dict["clarification_bonus"] = 2
            else:
                self.state_dict["clarification_bonus"] = max(
                    0, self.state_dict["clarification_bonus"] - 1
                )

        self._clip_relationships()

        self.history.append(
            {
                "step": self.state_dict["current_step"],
                "action": action_type,
                "reward": round(reward, 3),
            }
        )
        self.history = self.history[-self.history_length :]

        self.state_dict["done"] = self.state_dict["current_step"] >= self.state_dict["max_steps"]
        if not self.state_dict["done"]:
            next_index = min(
                self.state_dict["current_step"], self.state_dict["max_steps"] - 1
            )
            next_context = self.episode_plan["contexts"][next_index]
            self.state_dict["current_context"] = next_context
            self.state_dict["current_request"] = next_context["request"]

        self.state_dict["current_message"] = self._build_message()

        info = {
            "reward_components": reward_components,
            "drift_events": drift_effect["applied"],
            "delayed_penalties": delayed_effect["applied"],
        }
        if error is not None:
            info["error"] = error
        self.state_dict["last_info"] = info
        return self.state(), reward, self.state_dict["done"], info

    def state(self):
        context = self.state_dict.get("current_context", {})
        commitments = [
            {
                "remaining_steps": item["remaining_steps"],
                "impact": round(item["impact"], 2),
            }
            for item in self.state_dict.get("pending_commitments", [])
        ]

        return {
            "step": self.state_dict.get("current_step", 0),
            "request": self.state_dict.get("current_request", ""),
            "message": self.state_dict.get("current_message", ""),
            "visible_preferences": self.state_dict.get("preferences", []),
            "relationships": self.state_dict.get("relationships", {}),
            "history": list(self.history),
            "urgency": round(context.get("urgency", 0.0), 2),
            "impact": round(context.get("impact", 0.0), 2),
            "reversibility": round(context.get("reversibility", 0.0), 2),
            "delegation_feasibility": round(
                context.get("delegation_feasibility", 0.0), 2
            ),
            "commitments": commitments,
            "commitment_load": round(
                sum(
                    item["impact"]
                    for item in self.state_dict.get("pending_commitments", [])
                ),
                2,
            ),
        }

    def debug_state(self):
        return {
            "current_step": self.state_dict.get("current_step", 0),
            "true_weights": dict(self.state_dict.get("true_weights", {})),
            "drift_events": list(self.episode_plan.get("drift_events", []))
            if self.episode_plan
            else [],
            "pending_penalties": list(self.state_dict.get("pending_penalties", [])),
            "ask_count": self.state_dict.get("ask_count", 0),
            "config": dict(self.run_config),
        }

    def _generate_episode(self, scenario_index=None, config=None):
        config = config or {}
        if scenario_index in SCENARIO_LEVELS:
            difficulty = SCENARIO_LEVELS[scenario_index]
        else:
            difficulty = self.rng.choice(list(SCENARIO_PROFILES))

        profile = SCENARIO_PROFILES[difficulty]
        max_steps = 20

        drift_steps = sorted([
            self.rng.randint(2, max_steps - 2),
            self.rng.randint(2, max_steps - 2),
        ])
        first_step = drift_steps[0]
        second_step = drift_steps[1]

        apply_first = first_step + self.rng.randint(1, 2) if self.rng.random() < 0.5 else first_step
        apply_second = second_step + self.rng.randint(1, 2) if self.rng.random() < 0.5 else second_step

        phantom_step = self.rng.randint(2, max_steps - 2) if self.rng.random() < 0.15 else None

        initial_true_weights = {
            "work": self.rng.uniform(0.72, 0.84),
            "family": 0.0,
        }
        initial_true_weights["family"] = 1.0 - initial_true_weights["work"]
        family_needed = (
            initial_true_weights["work"] - initial_true_weights["family"] + 0.12
        )
        first_event = {
            "step": apply_first,
            "delta": {
                "family": max(self.rng.uniform(0.34, 0.44), family_needed),
                "work": -0.10,
            },
            "kind": "family_surge",
        }

        after_first = self._simulate_weights(initial_true_weights, first_event["delta"])
        work_needed = after_first["family"] - after_first["work"] + 0.12
        second_event = {
            "step": apply_second,
            "delta": {
                "work": max(self.rng.uniform(0.34, 0.46), work_needed),
                "family": -0.10,
            },
            "kind": "work_rebound",
        }

        drift_events = [first_event, second_event]

        if difficulty == "hard" or config.get("force_conflict"):
            third_step = min(max_steps, second_step + 2)
            if third_step <= max_steps:
                drift_events.append(
                    {
                        "step": third_step,
                        "delta": {
                            "family": self.rng.uniform(0.18, 0.28),
                            "work": self.rng.uniform(0.10, 0.18),
                        },
                        "kind": "conflicting_pull",
                    }
                )

        weights = dict(initial_true_weights)
        for event in drift_events:
            weights = self._simulate_weights(weights, event["delta"])
            event["target"] = max(weights, key=weights.get)

        contexts = []
        for _ in range(max_steps):
            contexts.append(
                {
                    "request": self.rng.choice(REQUEST_LIBRARY),
                    "urgency": self.rng.uniform(0.25, 0.95),
                    "impact": self.rng.uniform(0.25, 0.95),
                    "reversibility": self.rng.uniform(0.15, 0.90),
                    "delegation_feasibility": self.rng.uniform(0.10, 0.90),
                }
            )

        false_signal_rate = profile["false_signal_rate"]
        conflict_signal_rate = profile["conflict_signal_rate"]
        if config.get("false_signal"):
            false_signal_rate = max(false_signal_rate, 0.35)
        if config.get("force_conflict"):
            conflict_signal_rate = max(conflict_signal_rate, 0.45)

        return {
            "difficulty": difficulty,
            "max_steps": max_steps,
            "initial_true_weights": initial_true_weights,
            "drift_events": drift_events,
            "first_drift_step": first_step,
            "drift_signal_steps": drift_steps,
            "phantom_step": phantom_step,
            "signal_noise": profile["signal_noise"],
            "missing_signal_rate": profile["missing_signal_rate"],
            "false_signal_rate": false_signal_rate,
            "conflict_signal_rate": conflict_signal_rate,
            "contexts": contexts,
        }

    def _simulate_weights(self, weights, delta):
        updated = dict(weights)
        for key, value in delta.items():
            updated[key] = max(0.05, updated[key] + value)
        total = sum(updated.values())
        for key in updated:
            updated[key] /= total
        return updated

    def _apply_drift(self, step):
        applied = []
        for event in self.episode_plan["drift_events"]:
            if event["step"] != step or step in self.state_dict["applied_drift_steps"]:
                continue

            self.state_dict["true_weights"] = self._simulate_weights(
                self.state_dict["true_weights"], event["delta"]
            )
            self.state_dict["applied_drift_steps"].append(step)
            applied.append(event)
        return {"applied": applied}

    def _apply_pending_penalties(self, step):
        remaining = []
        applied = []
        reward_penalty = 0.0
        for penalty in self.state_dict["pending_penalties"]:
            if penalty["apply_at"] == step:
                reward_penalty += penalty["value"]
                applied.append(penalty)
            else:
                remaining.append(penalty)
        self.state_dict["pending_penalties"] = remaining
        return {"reward_penalty": reward_penalty, "applied": applied}

    def _advance_commitments(self):
        pending = []
        family_drag = 0.0
        boss_drag = 0.0
        reward_penalty = 0.0
        family_weight = self.state_dict["true_weights"]["family"]

        for item in self.state_dict["pending_commitments"]:
            family_drag += 0.025 * item["impact"]
            reward_penalty -= 0.09 * item["impact"] * family_weight
            item["remaining_steps"] -= 1
            if item["remaining_steps"] > 0:
                pending.append(item)
            else:
                boss_drag += 0.015 * item["urgency"]

        self.state_dict["pending_commitments"] = pending
        self.state_dict["relationships"]["family"] -= family_drag
        self.state_dict["relationships"]["boss"] -= boss_drag
        return {
            "reward_penalty": reward_penalty,
            "family_drag": family_drag,
            "boss_drag": boss_drag,
        }

    def _parse_action(self, action):
        if isinstance(action, str):
            try:
                parsed_action = json.loads(action)
            except Exception:
                return None, "", "Invalid JSON"
        elif isinstance(action, dict):
            parsed_action = action
        else:
            return None, "", "Invalid action format"

        action_type = parsed_action.get("type") or parsed_action.get("action", "")
        content = parsed_action.get("content", "")
        if action_type not in ACTIONS:
            return None, content, f"Invalid action type: {action_type}"
        return action_type, content, None

    def _compute_reward(self, action_type, content, commitment_penalty, delayed_penalty):
        del content
        context = self.state_dict["current_context"]
        work_weight = self.state_dict["true_weights"]["work"]
        family_weight = self.state_dict["true_weights"]["family"]
        urgency = context["urgency"]
        impact = context["impact"]
        reversibility = context["reversibility"]
        delegation = context["delegation_feasibility"]
        ambiguity = self._ambiguity_score()
        commitment_load = sum(
            item["impact"] for item in self.state_dict["pending_commitments"]
        )

        if action_type == "ACCEPT":
            alignment = (
                work_weight * (0.55 + 0.35 * urgency + 0.10 * impact)
                - family_weight
                * (0.34 + 0.45 * impact + 0.20 * (1.0 - reversibility))
            )
            action_specific = -0.10 * commitment_load * family_weight
        elif action_type == "REJECT":
            alignment = (
                family_weight * (0.56 + 0.32 * impact + 0.10 * (1.0 - reversibility))
                - work_weight * (0.34 + 0.42 * urgency)
            )
            action_specific = 0.02 if impact > 0.6 else 0.0
        elif action_type == "PROPOSE_RESCHEDULE":
            alignment = (
                work_weight * (0.18 + 0.28 * reversibility)
                + family_weight * (0.22 + 0.30 * reversibility + 0.10 * impact)
                - 0.18 * urgency
            )
            action_specific = 0.14 if self.state_dict["pending_commitments"] else 0.02
        elif action_type == "DELEGATE":
            alignment = (
                work_weight * (0.16 + 0.44 * delegation)
                + family_weight * (0.10 + 0.26 * delegation)
                - 0.15 * impact
            )
            action_specific = 0.10 if delegation > 0.55 else -0.08
        elif action_type == "ESCALATE":
            alignment = (
                0.08
                + 0.38 * ambiguity
                + 0.18 * impact * (1.0 - reversibility)
                - 0.12 * urgency
            )
            action_specific = 0.06 if ambiguity > 0.45 else -0.05
        elif action_type == "ASK_CLARIFICATION":
            alignment = (
                -0.08
                + 0.52 * ambiguity * impact
                + 0.12 * (1.0 - abs(work_weight - family_weight))
            )
            action_specific = 0.0
        elif action_type == "DRAFT_MESSAGE":
            alignment = -0.18 - 0.12 * urgency
            action_specific = -0.10
        else:
            alignment = -0.30
            action_specific = 0.0

        repetition_penalty = self._repetition_penalty(action_type)
        passive_penalty = -0.12 if action_type == "DRAFT_MESSAGE" else 0.0
        relationship_penalty = self._relationship_pressure_penalty(action_type)
        ask_penalty = self._ask_penalty(action_type)

        reward = (
            alignment
            + action_specific
            + repetition_penalty
            + passive_penalty
            + relationship_penalty
            + commitment_penalty
            + delayed_penalty
            + ask_penalty
        )
        reward = max(min(reward, 1.5), -1.0)

        return reward, {
            "alignment": round(alignment, 3),
            "action_specific": round(action_specific, 3),
            "repetition_penalty": round(repetition_penalty, 3),
            "passive_penalty": round(passive_penalty, 3),
            "relationship_penalty": round(relationship_penalty, 3),
            "commitment_pressure": round(commitment_penalty, 3),
            "delayed_penalty": round(delayed_penalty, 3),
            "ask_penalty": round(ask_penalty, 3),
        }

    def _ask_penalty(self, action_type):
        if action_type != "ASK_CLARIFICATION":
            return 0.0
        ask_count = self.state_dict["ask_count"]
        if ask_count == 1:
            return -0.10
        if ask_count == 2:
            return -0.25
        return -0.50

    def _ambiguity_score(self):
        context = self.state_dict["current_context"]
        weight_gap = abs(
            self.state_dict["true_weights"]["work"]
            - self.state_dict["true_weights"]["family"]
        )
        context_uncertainty = (
            0.25 * (1.0 - context["reversibility"])
            + 0.15 * context["impact"]
            + 0.08 * self.episode_plan["signal_noise"]
        )
        ambiguity = 0.60 * (1.0 - weight_gap) + context_uncertainty
        return max(0.0, min(1.0, ambiguity))

    def _relationship_pressure_penalty(self, action_type):
        projected = dict(self.state_dict["relationships"])
        delta_boss, delta_family = self._relationship_delta(action_type)
        projected["boss"] += delta_boss
        projected["family"] += delta_family
        deficit = max(0.0, 0.32 - projected["boss"]) + max(
            0.0, 0.32 - projected["family"]
        )
        return -0.28 * deficit

    def _repetition_penalty(self, action_type):
        repeated = 0
        for previous_action in reversed(self.action_window):
            if previous_action == action_type:
                repeated += 1
            else:
                break
        if repeated < 2:
            return 0.0
        return -0.12 * (repeated - 1)

    def _apply_action_effects(self, action_type):
        context = self.state_dict["current_context"]
        if action_type == "ACCEPT":
            self.state_dict["pending_commitments"].append(
                {
                    "remaining_steps": 2 + int(context["urgency"] > 0.7),
                    "impact": context["impact"],
                    "urgency": context["urgency"],
                }
            )
        elif (
            action_type == "PROPOSE_RESCHEDULE"
            and self.state_dict["pending_commitments"]
        ):
            resolved = self.state_dict["pending_commitments"].pop(0)
            if resolved["impact"] > 0.55:
                self.state_dict["relationships"]["boss"] -= 0.01
        elif action_type == "DELEGATE" and self.state_dict["pending_commitments"]:
            resolved = self.state_dict["pending_commitments"].pop(0)
            self.state_dict["relationships"]["boss"] += (
                0.01 * context["delegation_feasibility"]
            )
            self.state_dict["relationships"]["family"] += 0.01 * resolved["impact"]

    def _apply_action_relationships(self, action_type):
        delta_boss, delta_family = self._relationship_delta(action_type)
        self.state_dict["relationships"]["boss"] += delta_boss
        self.state_dict["relationships"]["family"] += delta_family

    def _relationship_delta(self, action_type):
        context = self.state_dict["current_context"]
        urgency = context["urgency"]
        impact = context["impact"]
        reversibility = context["reversibility"]
        delegation = context["delegation_feasibility"]
        ambiguity = self._ambiguity_score()

        if action_type == "ACCEPT":
            return (
                0.04 + 0.06 * urgency,
                -(0.05 + 0.09 * impact + 0.03 * (1.0 - reversibility)),
            )
        if action_type == "REJECT":
            return (-(0.05 + 0.08 * urgency), 0.03 + 0.06 * impact)
        if action_type == "PROPOSE_RESCHEDULE":
            return (-(0.01 + 0.02 * urgency), 0.02 + 0.05 * reversibility)
        if action_type == "DELEGATE":
            return (
                0.01 + 0.04 * delegation,
                -(0.01 + 0.04 * impact * (1.0 - delegation)),
            )
        if action_type == "ESCALATE":
            return (-(0.02 - 0.01 * ambiguity), -(0.01 - 0.01 * ambiguity))
        if action_type == "ASK_CLARIFICATION":
            return (-0.05 - 0.02 * urgency, 0.01 * ambiguity)
        if action_type == "DRAFT_MESSAGE":
            return (-0.04, -0.04)
        return (-0.03, -0.03)

    def _schedule_delayed_consequence(self, action_type):
        context = self.state_dict["current_context"]
        work_weight = self.state_dict["true_weights"]["work"]
        family_weight = self.state_dict["true_weights"]["family"]
        apply_at = self.state_dict["current_step"] + 2
        if apply_at > self.state_dict["max_steps"]:
            return

        penalty = None
        source = None

        if (
            action_type == "ACCEPT"
            and family_weight > work_weight
            and context["impact"] > 0.45
        ):
            penalty = -(0.18 + 0.30 * context["impact"] * family_weight)
            source = "stale_accept"
        elif (
            action_type == "REJECT"
            and work_weight > family_weight
            and context["urgency"] > 0.45
        ):
            penalty = -(0.18 + 0.28 * context["urgency"] * work_weight)
            source = "missed_work"
        elif action_type == "DRAFT_MESSAGE" and context["urgency"] > 0.60:
            penalty = -0.22
            source = "stalling"
        elif (
            action_type == "PROPOSE_RESCHEDULE"
            and context["reversibility"] < 0.35
            and context["urgency"] > 0.65
        ):
            penalty = -0.20
            source = "false_recovery"

        if penalty is None:
            return

        self.state_dict["pending_penalties"].append(
            {
                "apply_at": apply_at,
                "value": round(max(-0.50, penalty), 3),
                "source": source,
            }
        )

    def _apply_repetition_tracking(self, action_type):
        self.action_window.append(action_type)
        self.action_window = self.action_window[-self.history_length :]

    def _clip_relationships(self):
        for key, value in self.state_dict["relationships"].items():
            self.state_dict["relationships"][key] = max(0.0, min(1.0, value))

    def _build_message(self):
        opener = self.rng.choice(NEUTRAL_OPENERS)
        cue_type = self._select_public_signal_type()

        if cue_type == "family":
            cue = self.rng.choice(FAMILY_CUES)
        elif cue_type == "work":
            cue = self.rng.choice(WORK_CUES)
        elif cue_type == "conflict":
            cue = self.rng.choice(CONFLICT_CUES)
        else:
            cue = self.rng.choice(NEUTRAL_OPENERS)

        if self.state_dict["pending_commitments"] and self.rng.random() < 0.5:
            return f"{opener} {cue} {self.rng.choice(COMMITMENT_NOTES)}"
        return f"{opener} {cue}"

    def _select_public_signal_type(self):
        step = self.state_dict["current_step"]
        dominant = max(self.state_dict["true_weights"], key=self.state_dict["true_weights"].get)
        opposite = "family" if dominant == "work" else "work"
        first_drift_step = self.episode_plan["first_drift_step"]
        drift_steps = set(self.episode_plan.get("drift_signal_steps", [first_drift_step]))
        phantom_step = self.episode_plan.get("phantom_step")

        missing_rate = self.episode_plan["missing_signal_rate"]
        false_rate = self.episode_plan["false_signal_rate"]
        conflict_rate = self.episode_plan["conflict_signal_rate"]

        if self.state_dict["clarification_bonus"] > 0:
            missing_rate *= 0.35
            false_rate *= 0.25
            conflict_rate *= 0.50

        if self.run_config.get("force_conflict") and step in drift_steps:
            return "conflict"

        if step == phantom_step:
            return opposite

        if step < first_drift_step and self.rng.random() < 0.20:
            return opposite

        if step in drift_steps and self.rng.random() < 0.30:
            return "neutral"

        if self.rng.random() < 0.2:
            return opposite

        if self.run_config.get("delay_drift") and step in drift_steps:
            if self.rng.random() < 0.55 and self.state_dict["clarification_bonus"] == 0:
                return "neutral"

        roll = self.rng.random()
        if roll < missing_rate:
            return "neutral"
        if roll < missing_rate + false_rate:
            return opposite
        if roll < missing_rate + false_rate + conflict_rate:
            return "conflict"
        return dominant
