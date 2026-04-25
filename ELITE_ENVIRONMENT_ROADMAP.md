# StaleMind Elite Environment Roadmap

## North Star

Force the agent through this loop:

1. Form a belief
2. Notice evidence mismatch
3. Update belief under uncertainty
4. Change action
5. Recover from earlier mistakes

## Phase 1: Destroy Shortcuts

Files:
- `env/environment.py`
- `openenv.yaml`

Shipped in this pass:
- Continuous latent `work_weight` / `family_weight`
- Ambiguous, missing, false, and conflicting public signals
- `ASK_CLARIFICATION` action
- Semantic decision variables: `urgency`, `impact`, `reversibility`, `delegation_feasibility`
- Commitment load with future consequences

Next:
- Add explicit false-positive drift episodes
- Add neutral messages that still mention both work and family
- Add second drift event or partial reversals

## Phase 2: Deepen World Dynamics

Files:
- `env/environment.py`
- `validate_env_refactor.py` or a new validation script

Next:
- Multi-drift episodes
- Reversible drift
- Conflicting stakeholder pressure
- Delayed outcomes tied to commitments
- Better reward for calibrated information gathering

## Phase 3: Make Belief Visible

Files:
- `app.py`
- `main.py`

Shipped in this pass:
- Belief compass
- Thought stream
- Timeline with reward and commitment load

Next:
- Interactive "break the AI" controls
- Failure / recovery presets
- Uncertainty styling in the UI
- Side-by-side hidden truth vs inferred belief for judge mode only

## Phase 4: Training Alignment

Files:
- `StaleMind_GRPO_Training.ipynb`

Next:
- Remove notebook keyword heuristics
- Train on belief-state trajectories, not single-step lexical triggers
- Add curriculum:
  - clean drift
  - delayed drift
  - false signals
  - conflicting signals
  - multi-drift episodes
