# StaleMind: System Architecture & Technical Breakdown

This document provides a comprehensive, technically rigorous system-level breakdown of the **StaleMind** repository. It serves as a definitive guide for hackathon judges, senior engineers, and potential contributors.

---

## 1. 🔍 Project Overview

* **What is StaleMind?**
  StaleMind is a multi-step, Partially Observable Markov Decision Process (POMDP) Reinforcement Learning environment built to evaluate Large Language Models (LLMs) and autonomous agents.
* **Core idea in 1–2 lines:** 
  It tests an AI agent's ability to abandon explicit, stale system instructions when subtle environmental context implies that the user's true underlying preferences have shifted.
* **Real-world use case:** 
  Executive AI assistants, autonomous customer service agents, and long-running autonomous workflows where ground truth changes mid-operation without the user explicitly stopping to update the AI's prompt.
* **Target users:** 
  AI safety researchers, prompt engineers, and developers building autonomous agents who need to benchmark their models against "Contextual Drift."

---

## 2. 🧩 Problem Statement

* **What exact problem is being solved?**
  "Instruction adherence" is currently considered the gold standard for LLMs. However, strict instruction adherence becomes dangerous when the operating context changes but the prompt does not. StaleMind exposes the failure mode where agents blindly execute out-of-date instructions (e.g., prioritizing work over family) despite clear contextual evidence that the situation has changed (e.g., a family emergency).
* **Why is this problem important?**
  As AI transitions from single-turn chat to continuous autonomous operation, models must learn to recognize when their foundational assumptions have become stale to avoid executing catastrophic actions confidently.
* **Current limitations in existing solutions:**
  Standard RL environments (like OpenAI Gym or standard OpenEnv tasks) and static benchmarks (like MMLU) test static rule-following. They explicitly broadcast state changes. StaleMind intentionally hides the state change and only provides semantic clues.

---

## 3. 🏗️ System Architecture

* **High-level architecture diagram (Text Description):**
  The system is a unified monolithic container operating an ASGI server (Uvicorn). The server hosts a primary FastAPI application serving REST endpoints (`/reset`, `/step`, `/state`). Mounted directly onto the root (`/`) of this FastAPI instance is a Gradio application for visual demonstration.
* **Components/modules:**
  * **Core Logic (`env/environment.py`)**: The `StaleMindEnv` class. Manages episode state, relationship ledgers, drift triggering, and reward calculation.
  * **Backend API (`main.py`)**: A FastAPI wrapper that maps stateless HTTP requests to isolated, session-based environment instances.
  * **Frontend (`app.py`)**: A Gradio web interface providing a visual control panel to run side-by-side agent simulations.
  * **Validation Scripts (`capture_failure.py`, `compare_agents.py`)**: Standalone scripts that execute simulated or actual LLM interactions against the API.
* **How data flows through the system:**
  1. Client sends an HTTP POST to `/reset` with a `session_id`.
  2. `main.py` allocates a new `StaleMindEnv` in memory and returns an initial JSON observation.
  3. Client computes an action (via LLM or hardcoded logic) and sends HTTP POST to `/step`.
  4. `main.py` routes the action to the correct environment session.
  5. The environment computes relationships, checks for drift, and calculates rewards.
  6. The updated observation and reward are returned as JSON to the client.

---

## 4. ⚙️ Core Features (From Code, Not Assumptions)

### Hidden Preference Drift
* **What it does:** Mid-episode, the ground truth of the user's priority flips (from "work > family" to "family > work").
* **Where it exists:** `env/environment.py` (`step()` method).
* **How it works:** At a specific step index defined by the scenario difficulty (e.g., Step 5 for Medium), `self.state_dict["true_preferences"]` is mutated. Crucially, `self.state_dict["preferences"]` (the visible prompt) is intentionally left unmodified.

### Soft Semantic Signaling
* **What it does:** Injects natural language clues into the observation space when drift occurs.
* **Where it exists:** `env/environment.py` (`state()` method).
* **How it works:** If `drift_triggered` is True, a string like *"Your son has an event today."* is concatenated to the standard observation message. The agent must parse this NLP signal to deduce the state change.

### Relationship Ledger System
* **What it does:** Tracks long-term consequences of actions across the 10-step episode.
* **Where it exists:** `env/environment.py` (`step()` method).
* **How it works:** Actions deduct points from continuous tracking variables (e.g., REJECT penalizes the "boss" score by -0.1). The overall step reward is boosted by the average health of both relationships, penalizing agents that greedily optimize one preference while letting the other relationship collapse.

### Concurrency & Session Management
* **What it does:** Allows multiple simultaneous evaluations against the API without state corruption.
* **Where it exists:** `main.py`.
* **How it works:** An in-memory dictionary `envs: Dict[str, DriftGym]` maps unique `session_id` strings to dedicated environment instances. 

---

## 5. 🤖 AI / LLM Integration

* **Which models are used:** 
  * HuggingFace Inference API (`Qwen/Qwen2.5-7B-Instruct`) in `capture_failure.py`.
  * `meta-llama/Llama-3.2-3B-Instruct` in `preflight.py`.
* **Prompt engineering strategy:**
  The agent is rigidly prompted to adhere to `visible_preferences`. It is told: *"You MUST follow the stated preferences. Do not override them."* This intentionally sets up the model to fail when drift occurs, testing its ability to break rules when context demands it.
* **Input → Output pipeline:**
  The LLM receives a serialized string of the observation JSON. It is forced to output a structured JSON containing `"type"` (the action), `"content"`, and `"reasoning"`.
* **Guardrails / sanitization logic:**
  In `capture_failure.py`, if the LLM fails to generate valid JSON or hallucinates an action outside the 6 allowed actions, a fallback block forces the action to `ACCEPT` and logs a parse error.

---

## 6. 🧪 Execution Flow (End-to-End)

*Tracing exactly what happens when a user opens the UI and runs a simulation:*

1. **Opens the UI**: The user visits the Hugging Face Space. Uvicorn routes the root `/` path to the mounted Gradio `app`. The browser renders the React-based Gradio frontend.
2. **Interacts with Mission Control**: User selects "Medium (drift at step 5)" and "Adaptive (drift-aware)" agent, then clicks "Run Episode".
3. **Submits a task**: Gradio triggers the `run_full_episode` python function.
   * `uuid.uuid4()` generates a unique `session_id`.
   * A synchronous internal `requests.post("http://127.0.0.1:7860/reset")` initializes the backend state.
4. **Agent Loop Execution**:
   * For 10 iterations, the `run_full_episode` function simulates the "Adaptive" agent using regex: `if any(w in msg for w in ["son", "event"...]) -> REJECT`.
   * It sends a `requests.post` to `/step` with the action payload.
   * FastAPI routes this to the specific `session_id`.
   * The environment computes the step, and the reward is captured.
5. **Receives output**: The 10-step log is formatted into ASCII text bars and rendered back into the Gradio Textbox on the frontend.

---

## 7. 🧱 Tech Stack

* **Languages**: Python 3.10
* **Frameworks**: FastAPI (Backend API), Gradio (Frontend Visualization)
* **Libraries**: `requests` (API querying), `pydantic` (Data validation/schemas), `matplotlib` & `numpy` (Chart generation in scripts)
* **Dev tools**: OpenEnv (Standardized RL environment metadata)
* **Deployment**: Dockerfile deploying directly to Hugging Face Spaces using Uvicorn ASGI server.

---

## 8. 🔐 Constraints & Limitations

* **Known bugs or incomplete areas**:
  * **Memory Leak Potential**: The `envs` dictionary in `main.py` stores sessions indefinitely. There is no TTL (Time To Live) or cleanup mechanism for abandoned `session_ids`. In a high-traffic environment, this will eventually exhaust RAM.
  * **Simulated vs Real Agents in UI**: The Gradio UI currently uses hardcoded if/else regex logic to represent the "Naive" and "Adaptive" agents. Real LLMs are only connected in the offline `capture_failure.py` script.
* **Performance bottlenecks**:
  * The Gradio UI executes synchronous `requests.post` calls to its own Uvicorn host. While functional, async `httpx` would be more performant and prevent thread-blocking under high concurrent load.

---

## 9. 🚀 Innovation & Differentiation

* **What makes StaleMind different from ChatGPT-style apps**:
  It is not a consumer app; it is an adversarial testing ground. It is designed specifically to induce catastrophic failure in models that over-index on system prompts.
* **What makes it different from standard RL benchmarks**:
  Standard environments (like CartPole or basic text games) explicitly update their state variables. StaleMind is a POMDP where the core variable (`true_preferences`) is invisible, requiring semantic reasoning over NLP text strings to deduce the state.
* **Why this matters**:
  It isolates "stubbornness" vs "adaptability" in LLMs, providing a metric for how much counter-evidence an agent requires before breaking a direct system instruction.

---

## 10. 🧠 Design Decisions

* **Unified API/UI Mounting (`gr.mount_gradio_app`)**:
  * **Why:** To avoid CORS nightmares, port conflicts in HuggingFace Spaces, and the complexity of managing multiple processes (e.g., Supervisord).
  * **Trade-off:** Couples the UI and API into a single Python process, meaning a heavy UI computation could theoretically block API request handling.
* **Continuous Reward Function**:
  * **Why:** Instead of a binary Pass/Fail, actions yield partial rewards (e.g., `PROPOSE_RESCHEDULE` yields 0.5 or 0.4).
  * **Trade-off:** Requires careful balancing to ensure agents cannot exploit compromise actions to achieve high scores without actually deducing the drift.

---

## 11. 📈 Future Scope

* **What can be improved**:
  * Implement an automated garbage collection routine in `main.py` to prune `session_ids` older than 1 hour.
  * Refactor `app.py` to use asynchronous HTTP requests (`httpx.AsyncClient`) for the simulation loop.
* **What features should be added next**:
  * Connect the Gradio UI directly to an LLM provider (via API key input) so users can watch real models fail or adapt live in the UI, rather than relying on the simulated regex agents.
  * Implement an "LLM-as-a-judge" module to generate dynamic, infinite variations of incoming scheduling conflicts and drift signals to prevent data contamination during training.
* **Scaling strategy**:
  * To train models (e.g., using TRL's `GRPOTrainer`), the environment should be decoupled from HTTP. Packaging `StaleMindEnv` into an installable PyPI library (`pip install stalemind`) would allow direct Python imports, completely removing network overhead for large-scale RLHF training.
