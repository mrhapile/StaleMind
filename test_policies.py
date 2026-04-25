import uuid
import json
from stalemind_learning import rollout_policy

policies = ["step_threshold", "keyword", "always_ask", "adaptive", "random"]

print("==== EVALUATING POLICIES ====")
for policy in policies:
    rewards = []
    scores = []
    for i in range(20):
        # Medium scenario with all adversarial flags enabled
        res = rollout_policy(
            scenario_index=1, 
            seed=i, 
            config={"false_signal": True, "delay_drift": True, "force_conflict": True}, 
            policy_kind=policy
        )
        rewards.append(res["final_reward"])
        scores.append(res["adaptation_score"])
    
    avg_reward = sum(rewards)/len(rewards)
    avg_score = sum(scores)/len(scores)
    print(f"{policy:15s} -> Avg Reward: {avg_reward:6.2f} | Avg Score: {avg_score:6.2f}")
