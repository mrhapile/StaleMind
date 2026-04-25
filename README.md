---
title: Stalemind
emoji: 🐨
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: A decision-making environment where agents fail under hidden
---
# StaleMind
An RL-style environment demonstrating how autonomous agents fail when their internal understanding of the world becomes stale.

## Problem Statement
Autonomous AI agents operate based on models of user preferences, environmental states, and explicit instructions. However, in the real world, human priorities and situations shift dynamically. When an agent's internal model is not continuously updated to reflect these hidden shifts, the agent will confidently execute actions that are technically correct according to past instructions but contextually disastrous in the present. Standard environments do not adequately test an agent's ability to detect and adapt to unannounced contextual drift.

## What is StaleMind
StaleMind is a multi-step OpenAI Gym-style environment designed to test agent adaptability against hidden state drift. It simulates an executive assistant scenario where the agent must manage scheduling conflicts between work obligations and family priorities. Mid-episode, the underlying ground truth of what the user prioritizes shifts (e.g., from "work > family" to "family > work"). The environment does not explicitly broadcast this change; instead, it surfaces soft signals in the observation space. The agent must interpret these signals, update its understanding, and adjust its actions accordingly.

## Environment Design
The environment is structured as a 10-step episode. Key features include:
- Multi-step interaction: Each episode spans a fixed length of 10 discrete steps.
- Hidden Ground Truth: The environment maintains a `true_preferences` state that the agent cannot directly view.
- Unannounced Drift: Based on a configurable scenario difficulty (easy, medium, hard), the `true_preferences` state changes at a specific step mid-episode.
- Scenario Configurations: The drift step and the clarity of the soft signals vary based on the selected difficulty.
- Relationship Ledger: The environment tracks relationship scores with both the "boss" and "family", which are updated based on the agent's actions.

## Action Space
The agent can choose from 6 discrete actions, each allowing for optional textual content (e.g., when proposing a reschedule):
1. ACCEPT: Accept the incoming request.
2. REJECT: Reject the incoming request.
3. ESCALATE: Ask for explicit human intervention.
4. PROPOSE_RESCHEDULE: Offer an alternative time (content payload parsed for keywords like "tomorrow" or "later").
5. DELEGATE: Assign the request to someone else.
6. DRAFT_MESSAGE: Draft a holding message without committing to a decision.

## Observation Space
At each step, the environment returns a structured observation dictionary containing:
- `step`: The current step number (0 to 10).
- `request`: The specific scheduling conflict or request presented to the agent.
- `message`: Contextual information from the environment. Once drift occurs, this message incorporates soft signals indicating a shift in priorities.
- `visible_preferences`: The agent's initial, static understanding of the user's preferences. This never updates, forcing the agent to rely on environmental context.
- `relationships`: A dictionary containing the current scores for the "boss" and "family" ledgers.

## Reward Function
The reward system is continuous (0.0 to 1.0) rather than purely binary, encouraging nuanced decision-making:
- Base Reward: Actions are scored against the hidden `true_preferences`. For example, if the true preference is family, REJECT yields 1.0, while ACCEPT yields 0.0. Compromise actions like PROPOSE_RESCHEDULE yield partial rewards (e.g., 0.7).
- Content Bonus: If the action is PROPOSE_RESCHEDULE, the environment parses the content payload. Using appropriate keywords grants an additional 0.1 bonus.
- Ledger Bonus: The overall reward is boosted by a fraction of the combined relationship scores, ensuring that long-term relationship management factors into immediate action evaluation.

## How It Works
1. Initialization: The environment resets, selecting a scenario and establishing the initial state.
2. Early Steps: The agent processes incoming requests. Actions aligned with the initial preferences yield high rewards.
3. Drift Event: At a pre-defined step, the hidden `true_preferences` flip.
4. Soft Signaling: The observation `message` begins to include subtle hints about the new priority (e.g., "Things at home are busy.").
5. Adaptation Phase: The agent must recognize the signal, infer the new priority, and change its action strategy.
6. Termination: The episode concludes after 10 steps.

## Example Interaction
- Step 1: Request is "boss schedules urgent call". True preference is work. Agent chooses ACCEPT. Reward is 1.0.
- Step 2-4: Similar requests. Agent continues to ACCEPT. Reward remains 1.0.
- Step 5 (Drift): True preference silently shifts to family. Observation message adds: "Things at home are busy."
- Step 6: Request is "boss schedules last-minute urgent meeting". Agent chooses ACCEPT (relying on stale preferences). Reward drops to 0.0.
- Step 7: Agent recognizes the signal and chooses PROPOSE_RESCHEDULE with content "Can we do this tomorrow?". Reward is approximately 0.8.

## Why This Project is Unique
Current RL and LLM-agent benchmarks primarily test an agent's ability to follow static instructions or achieve static goals. StaleMind isolates a critical failure mode in autonomous systems: the inability to detect when the context invalidates the initial prompt. By using soft signals and a relationship ledger instead of explicit state updates, StaleMind forces agents to demonstrate genuine contextual awareness and continuous model updating.

## Tech Stack
- Python 3
- FastAPI (for serving the environment as an accessible API)
- Uvicorn (ASGI server)
- Pydantic (data validation for the API endpoints)

## Project Structure
```text
StaleMind/
├── env/
│   ├── __init__.py
│   └── environment.py    (Core environment logic, state management, reward calculation)
├── main.py               (FastAPI server exposing the environment)
├── requirements.txt      (Dependencies)
├── .gitignore
└── README.md             (Project documentation)
```

## How to Run
1. Navigate to the project directory:
   `cd StaleMind`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the server:
   `uvicorn main:app --reload`
4. The API will be available at `http://127.0.0.1:8000`. You can interact with the environment via POST requests to `/reset` and `/step`, and GET requests to `/state`.

## Failure Case

When tested against `Qwen/Qwen2.5-7B-Instruct`, the model was instructed to follow the user's stated preferences (`work > family`). After the hidden drift event flipped the true preference to `family > work`, the model continued to ACCEPT work requests because its visible preferences were never updated. This produced a sharp reward drop from 1.0 to near 0.0 on every post-drift step.

Example from Medium scenario (drift at step 5):
```
Step  4 | ACCEPT  | reward: 1.00
Step  5 | ACCEPT  | reward: 0.03   <- drift happened, model did not adapt
Step  6 | ACCEPT  | reward: 0.03   | DRIFT ACTIVE | *** FAILURE ***
Step  7 | ACCEPT  | reward: 0.03   | DRIFT ACTIVE | *** FAILURE ***
```

The model's reasoning remained: "The user's stated preference is work > family, so I must accept the work request." This confirms the core thesis: an agent that trusts stale context will fail silently and confidently.

## Evaluation

Comparison of Naive (always ACCEPT) vs Adaptive (drift-aware) agents across all scenarios:

| Scenario | Naive Agent | Adaptive Agent | Improvement |
|----------|-------------|----------------|-------------|
| Easy     | 6.10        | 9.04           | +2.94       |
| Medium   | 6.10        | 9.03           | +2.93       |
| Hard     | 6.10        | 9.03           | +2.93       |

The adaptive agent detects soft drift signals in the observation message (keywords like "son", "home", "busy") and switches from ACCEPT to REJECT, maintaining high reward throughout the episode.

## Demo

Live environment API: https://mrhapile-stalemind.hf.space

Endpoints:
- `POST /reset` — Reset environment (optional: `{"scenario_index": 0|1|2}`)
- `POST /step` — Take action (`{"type": "ACCEPT", "content": ""}`)
- `GET /state` — Current observation

## Future Improvements
- Multi-Agent Scenarios: Introduce secondary agents that the primary agent can query to clarify ambiguous drift signals.
- Continuous Action Spaces: Expand the action payload to allow for fully generated text responses evaluated by an LLM-as-a-judge.
- Dynamic Scenarios: Implement an LLM-driven scenario generator to create infinite variations of drift signals and incoming requests.
- Advanced Ledger Dynamics: Make relationship scores decay over time or impact the difficulty of future requests.
