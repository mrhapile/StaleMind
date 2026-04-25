import random

ACTIONS = [
    "ACCEPT",
    "REJECT",
    "ESCALATE",
    "PROPOSE_RESCHEDULE",
    "DELEGATE",
    "DRAFT_MESSAGE"
]

SCENARIOS = [
    {
        "difficulty": "easy",
        "drift_step": 7,
        "drift_signal": "Your son really needs you today. He has been asking for you all week.",
        "request": "boss schedules urgent call at 7pm"
    },
    {
        "difficulty": "medium",
        "drift_step": 5,
        "drift_signal": "Your son has an event today.",
        "request": "boss schedules urgent call during family time"
    },
    {
        "difficulty": "hard",
        "drift_step": 3,
        "drift_signal": "Things at home are busy.",
        "request": "boss schedules last-minute urgent meeting"
    }
]

class StaleMindEnv:
    def __init__(self):
        self.state_dict = {
            "current_step": 0,
            "max_steps": 10,
            "done": False,
            "current_request": "boss schedules urgent call"
        }

    def reset(self, scenario_index=None):
        if scenario_index is not None and scenario_index in [0, 1, 2]:
            self.scenario = SCENARIOS[scenario_index]
        else:
            self.scenario = random.choice(SCENARIOS)

        self.state_dict = {
            "current_step": 0,
            "max_steps": 10,
            "done": False,
            "current_request": self.scenario.get("request", "boss schedules urgent call"),
            "preferences": ["work > family"],
            "true_preferences": ["work > family"],
            "drift_triggered": False,
            "drift_step": self.scenario["drift_step"],
            "drift_signal": self.scenario["drift_signal"],
            "last_action": None,
            "relationships": {
                "boss": 0.5,
                "family": 0.5
            }
        }
        return self.state(), {}

    def step(self, action):
        if self.state_dict["done"]:
            return None, 0.0, True, {}

        self.state_dict["current_step"] += 1

        if self.state_dict["current_step"] == self.state_dict["drift_step"] and not self.state_dict["drift_triggered"]:
            self.state_dict["true_preferences"] = ["family > work"]
            self.state_dict["drift_triggered"] = True

        # FIX: Action Parsing Robustness
        if isinstance(action, str):
            try:
                import json
                parsed_action = json.loads(action)
                action_type = parsed_action.get("type") or parsed_action.get("action", "")
                content = parsed_action.get("content", "")
            except Exception:
                # FIX: Remove Episode Termination on Invalid JSON
                return self.state(), -2.0, False, {"error": "Invalid JSON"}
        elif isinstance(action, dict):
            action_type = action.get("type") or action.get("action", "")
            content = action.get("content", "")
        else:
            # FIX: Remove Episode Termination on Invalid format
            return self.state(), -2.0, False, {"error": "Invalid action format"}

        if action_type not in ACTIONS:
            # FIX: Remove Episode Termination on Invalid type
            return self.state(), -2.0, False, {"error": f"Invalid action type: {action_type}"}

        reward = self._compute_reward(action_type, content)

        if action_type == "REJECT":
            self.state_dict["relationships"]["boss"] -= 0.1

        if action_type == "ACCEPT":
            self.state_dict["relationships"]["family"] -= 0.1

        if action_type == "PROPOSE_RESCHEDULE":
            self.state_dict["relationships"]["boss"] -= 0.05
            self.state_dict["relationships"]["family"] -= 0.05

        for k in self.state_dict["relationships"]:
            self.state_dict["relationships"][k] = max(0.0, min(1.0, self.state_dict["relationships"][k]))

        if self.state_dict["current_step"] >= self.state_dict["max_steps"]:
            self.state_dict["done"] = True

        return self.state(), reward, self.state_dict["done"], {}

    def state(self):
        message = "You have a scheduling conflict."
        if self.state_dict.get("drift_triggered", False):
            message += " " + self.state_dict.get("drift_signal", "")

        return {
            "step": self.state_dict.get("current_step", 0),
            "request": self.state_dict.get("current_request", ""),
            "message": message,
            "visible_preferences": self.state_dict.get("preferences", []),
            "relationships": self.state_dict.get("relationships", {})
        }

    def _compute_reward(self, action_type, content):

        reward = 0.0
        true_pref = self.state_dict["true_preferences"][0]

        if true_pref == "work > family":
            if action_type == "ACCEPT":
                reward = 1.0
            elif action_type == "PROPOSE_RESCHEDULE":
                reward = 0.5
            elif action_type == "ESCALATE":
                reward = 0.5
            elif action_type == "DELEGATE":
                reward = 0.6
            elif action_type == "DRAFT_MESSAGE":
                reward = 0.4
            else:
                reward = 0.0

        elif true_pref == "family > work":
            if action_type == "REJECT":
                reward = 1.0
            elif action_type == "PROPOSE_RESCHEDULE":
                reward = 0.4
            elif action_type == "DRAFT_MESSAGE":
                reward = 0.6
            elif action_type == "ESCALATE":
                reward = 0.5
            elif action_type == "DELEGATE":
                reward = 0.4
            else:
                reward = 0.0

        # FIX: Weak negative feedback
        if reward == 0.0:
            reward = -0.2

        # FIX: Reward signal saturation
        reward *= 0.8

        content_lower = content.lower()
        reschedule_keywords = ["later", "tomorrow", "after", "reschedule", "move", "shift"]

        # IMPROVED: Content-Based Reward
        if action_type == "PROPOSE_RESCHEDULE":
            if any(word in content_lower for word in reschedule_keywords):
                reward += 0.2

        # FIX: Add Drift Awareness Reward
        if self.state_dict["drift_triggered"]:
            if "son" in content_lower or "family" in content_lower:
                reward += 0.2

        # FIX: Prevent Exploit: Always ACCEPT
        if action_type == "ACCEPT" and self.state_dict["drift_triggered"]:
            reward -= 0.2

        # FIX: No reward for consistency over time
        prev_action = self.state_dict.get("last_action")
        if prev_action == action_type:
            reward += 0.05
        self.state_dict["last_action"] = action_type

        boss_score = self.state_dict["relationships"]["boss"]
        family_score = self.state_dict["relationships"]["family"]

        # FIX: Relationship Penalty (CRITICAL)
        if boss_score < 0.2 or family_score < 0.2:
            reward -= 0.3

        relationship_bonus = 0.1 * (boss_score + family_score) / 2
        reward += relationship_bonus

        # FIX: Keep Reward Bounded
        reward = max(-2.0, min(1.0, reward))

        return reward
