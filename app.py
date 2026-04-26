import uuid
import requests
import random

try:
    from fastapi.middleware.cors import CORSMiddleware
    from main import app as api
except ImportError:  # pragma: no cover
    CORSMiddleware = None
    api = None

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "http://127.0.0.1:7860"

def create_state():
    return {
        "belief_work": 0.5,
        "belief_family": 0.5,
        "history": [],
        "last_action": None,
        "last_reward": 0.0,
        "current_obs": "Wait... The story hasn't started yet. Initialize the engine.",
        "done": False,
        "session_id": str(uuid.uuid4())
    }

def normalize_belief(bw, bf):
    total = bw + bf
    if total == 0:
        return 0.5, 0.5
    return bw / total, bf / total

def reset_fn(state=None):
    if state is None:
        state = create_state()
    else:
        state = create_state()
        
    try:
        scenario_idx = random.randint(0, 2) # Ensuring scenarios vary
        res = requests.post(BASE_URL + "/reset", json={"session_id": state["session_id"], "scenario_index": scenario_idx}).json()
        state["current_obs"] = res.get("observation", {}).get("message", "System reset.")
    except Exception as e:
        state["current_obs"] = "Could not connect to backend API. Please make sure the server is initialized properly or hit REBOOT."
    return state, *render_all(state, is_reset=True)

def step_fn(action, state):
    if state["done"]:
        return state, *render_all(state)

    try:
        # Standardized flatten request
        res = requests.post(
            BASE_URL + "/step", 
            json={"type": action, "session_id": state["session_id"], "content": ""}
        ).json()
        
        # Robust parsing of standardized response
        reward_data = res.get("reward", {})
        if isinstance(reward_data, dict):
            reward = float(reward_data.get("alignment", 0))
        else:
            reward = float(reward_data)

        obs_data = res.get("observation", {})
        if isinstance(obs_data, dict):
            message = obs_data.get("message", "No response")
            raw_obs = obs_data.get("raw", obs_data)
        else:
            message = str(obs_data)
            raw_obs = {}

        info = res.get("info", {})
        
        state["last_action"] = action
        state["last_reward"] = reward
        state["current_obs"] = message # Store the narrative message
        state["done"] = res.get("done", False)

        # For backward compatibility with the logging logic expectations
        current_obs_dict = raw_obs if isinstance(raw_obs, dict) else {}

        prev_bw = state["belief_work"]
        prev_bf = state["belief_family"]

        # Phase 3 Adaptive Intelligence Logic
        update_strength = 0.15
        if action == "ACCEPT":
            state["belief_work"] += reward * update_strength
        elif action == "REJECT":
            state["belief_family"] += reward * update_strength
        elif action == "ASK_CLARIFICATION":
            # Pull beliefs toward center (0.5) to show increased uncertainty
            state["belief_work"] = 0.5 + (state["belief_work"] - 0.5) * 0.85
            state["belief_family"] = 0.5 + (state["belief_family"] - 0.5) * 0.85
            # Small random jitter for visual feedback
            state["belief_work"] += random.uniform(-0.02, 0.02)
            state["belief_family"] += random.uniform(-0.02, 0.02)

        state["belief_work"] *= 0.95
        state["belief_family"] *= 0.95
        state["belief_work"], state["belief_family"] = normalize_belief(max(0, state["belief_work"]), max(0, state["belief_family"]))

        # Build rich log entry from API response
        rc = info.get("reward_components", {})
        drift_events = info.get("drift_events", [])
        delayed_penalties = info.get("delayed_penalties", [])
        relationships = current_obs_dict.get("relationships", {})
        urgency = current_obs_dict.get("urgency", 0)
        impact = current_obs_dict.get("impact", 0)
        reversibility = current_obs_dict.get("reversibility", 0)
        delegation = current_obs_dict.get("delegation_feasibility", 0)
        commitment_load = current_obs_dict.get("commitment_load", 0)
        commitments = current_obs_dict.get("commitments", [])
        env_history = current_obs_dict.get("history", [])
        visible_prefs = current_obs_dict.get("visible_preferences", [])
        
        state["history"].append({
            "step": len(state["history"]) + 1,
            "action": action,
            "reasoning": info.get("reasoning", "The engine is processed this action based on current staleness vectors."),
            "reward": reward,
            "prev_bw": prev_bw,
            "prev_bf": prev_bf,
            "reward_components": rc,
            "drift_events": drift_events,
            "delayed_penalties": delayed_penalties,
            "relationships": relationships,
            "urgency": urgency,
            "impact": impact,
            "reversibility": reversibility,
            "delegation": delegation,
            "commitment_load": commitment_load,
            "commitments": commitments,
            "env_history": env_history,
            "visible_prefs": visible_prefs,
            "belief_work": state["belief_work"],
            "belief_family": state["belief_family"],
        })

    except Exception as e:
        print("API Step Failed:", e)

    return state, *render_all(state, is_reset=False)


# --- RENDER COMPONENTS ---

def render_situation(state):
    obs = state.get("current_obs", "System standby...")
    if isinstance(obs, dict):
        text = obs.get("message") or str(obs)
    else:
        text = str(obs)
    
    # CRITICAL: escape curly braces so f-string never crashes
    text = text.replace('{', '{{').replace('}', '}}')
    done_banner = ""
    if state.get("done"):
        done_banner = "<div style='color:#f59e0b; font-family:\"Share Tech Mono\",monospace; font-size:0.75rem; margin-bottom:0.5rem; letter-spacing:0.15em;'>⚡ EPISODE COMPLETE — HIT REBOOT TO RESET</div>"
    return f"""
    <div style='font-family: "Share Tech Mono", monospace; color: #38bdf8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;'>
        <div style='width: 6px; height: 6px; background: #38bdf8; border-radius: 50%; box-shadow: 0 0 10px #38bdf8; animation: pulseGlow 2s infinite;'></div>
        LIVE SCENARIO FEED
    </div>
    {done_banner}
    <div style='background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6)); 
                padding: 0.85rem 1rem; border-left: 4px solid #38bdf8;
                color:#f8fafc; font-size: 1rem; line-height: 1.5; font-weight: 500; font-family: "Chakra Petch", sans-serif;'>
        {text}
    </div>
    <style>@keyframes pulseGlow {{ 0% {{opacity:0.5;}} 50% {{opacity:1;}} 100% {{opacity:0.5;}} }}</style>
    """

def render_decision(state):
    if not state["history"]:
        return "<div style='text-align: center; color: #475569; padding: 2rem; font-family: \"Share Tech Mono\", monospace; font-size: 1.1rem; letter-spacing: 0.15em;'>[ SYSTEM STANDBY... ]</div>"

    action = state["last_action"]
    reward = state["last_reward"]

    if reward > 0:
        color = "#10b981"
        border_color = "#10b981"
        icon = "✔"
        label = "ALIGNMENT SECURED"
    else:
        color = "#ef4444"
        border_color = "#ef4444"
        icon = "✖"
        label = "CONFLICT DETECTED"

    shift_label = ""
    if len(state["history"]) > 1:
        prev_action = state["history"][-2]["action"]
        if prev_action != action:
            shift_label = f"<div style='color: #8b5cf6; font-size: 0.7rem; font-weight: bold; margin-bottom:0.5rem; letter-spacing: 0.15em; font-family: \"Share Tech Mono\", monospace;'>⚡ TACTICAL SHIFT: {prev_action} → {action}</div>"

    return f"""
    <div style='
        background: {color}20;
        border: 2px solid {border_color}66;
        border-left: 8px solid {border_color};
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        margin-top: 0.5rem;
    '>
        <div style='font-size: 2rem; color: {color}; flex-shrink: 0; text-shadow: 0 0 12px {color}88;'>{icon}</div>
        <div>
            {shift_label}
            <div style='font-family: "Share Tech Mono", monospace; color: #94a3b8; font-size: 0.75rem; letter-spacing: 0.2em; margin-bottom: 0.15rem;'>INFERENCE ENGINE OUTPUT</div>
            <div style='font-family: "Chakra Petch", sans-serif; font-size: 1.3rem; color: #f8fafc; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;'>{action}</div>
            <div style='color: {color}; font-weight: 700; font-size: 0.9rem; font-family: "Share Tech Mono", monospace; letter-spacing: 0.1em; margin-top: 0.25rem;'>{label} &nbsp;[{reward:+.2f}]</div>
        </div>
    </div>
    """

def render_meter(state):
    bw = state["belief_work"]
    bf = state["belief_family"]

    angle = (bf - bw) * 90
    
    if state["history"]:
        prev_bw = state["history"][-1].get("prev_bw", 0.5)
        prev_bf = state["history"][-1].get("prev_bf", 0.5)
        prev_angle = (prev_bf - prev_bw) * 90
    else:
        prev_angle = angle

    import string
    anim_name = f"swing_{''.join(random.choices(string.ascii_letters, k=6))}"
    
    if abs(bf - bw) < 0.2:
        subtitle = "Neutral / Uncertain"
    elif bw > bf:
        subtitle = "Leaning Work"
    else:
        subtitle = "Leaning Family"

    return f"""
    <style>
    @keyframes {anim_name} {{
        0% {{ transform: rotate({prev_angle}deg); }}
        100% {{ transform: rotate({angle}deg); }}
    }}
    </style>
    <div style="padding: 1.5rem; text-align: center; display: flex; flex-direction: column; align-items: center; background: radial-gradient(circle at 50% 55%, rgba(249, 115, 22, 0.22) 0%, transparent 85%);">
        <div style="color: #64748b; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; margin-bottom: 1.5rem; letter-spacing: 0.2em; text-transform: uppercase;">
            [{subtitle}]
        </div>
        <!-- COMPASS DIAL -->
        <div style="
            width: 500px; height: 250px;
            background: linear-gradient(180deg, rgba(30,41,59,0.6) 0%, rgba(2,6,23,0.85) 100%);
            border-radius: 250px 250px 0 0;
            position: relative;
            box-shadow: 
                0 20px 50px rgba(0,0,0,0.8),
                inset 0 3px 5px rgba(255,255,255,0.08),
                inset 0 0 30px rgba(0,0,0,0.6);
            border: 4px solid rgba(51,65,85,0.7);
            border-bottom: 3px solid rgba(2,6,23,0.5);

            display: flex; justify-content: center;
            backdrop-filter: blur(4px);
        ">
            <!-- Inner Recessed Track -->
            <div style="
                position: absolute; bottom: -8px; width: 460px; height: 230px;
                border-radius: 230px 230px 0 0;
                background: radial-gradient(ellipse at bottom, rgba(2,6,23,0.9) 0%, rgba(15,23,42,0.7) 100%);
                box-shadow: inset 0 15px 30px rgba(0,0,0,0.9);
                border-top: 1px solid rgba(255,255,255,0.04);
            ">
                <svg viewBox="0 0 200 100" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0;">
                    <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="4" stroke-dasharray="1 15.6" />
                    <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="rgba(255,255,255,0.02)" stroke-width="2" stroke-dasharray="1 3.1" />
                    <path d="M 10 100 A 90 90 0 0 1 80 38" fill="none" stroke="rgba(56,189,248,0.25)" stroke-width="12" />
                    <path d="M 120 38 A 90 90 0 0 1 190 100" fill="none" stroke="rgba(249,115,22,0.25)" stroke-width="12" />
                </svg>
                <div style="position: absolute; bottom: 30px; left: 30px; font-family: 'Share Tech Mono', monospace; font-size: 1rem; color: #38bdf8; text-shadow: 0 0 15px rgba(56,189,248,0.5);">[WRK]</div>
                <div style="position: absolute; bottom: 30px; right: 30px; font-family: 'Share Tech Mono', monospace; font-size: 1rem; color: #f97316; text-shadow: 0 0 15px rgba(249,115,22,0.5);">[FAM]</div>
            </div>
            <!-- Glare -->
            <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 500px; height: 125px; background: radial-gradient(ellipse at top, rgba(255,255,255,0.04) 0%, transparent 70%); border-radius: 250px 250px 0 0; pointer-events: none;"></div>
            <!-- Needle -->
            <div style="position: absolute; bottom: 0; left: 50%; width: 0; height: 0;">
                <div style="
                    position: absolute; bottom: 10px; left: -5px; width: 10px; height: 175px;
                    transform-origin: bottom center;
                    background: linear-gradient(to right, #64748b 0%, #e2e8f0 50%, #475569 100%);
                    border-radius: 10px 10px 3px 3px;
                    animation: {anim_name} 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
                    box-shadow: 4px 4px 10px rgba(0,0,0,0.9), inset 1px 0 2px rgba(255,255,255,0.5);
                ">
                    <div style="position: absolute; top: 0; left: 0; right: 0; height: 35px; background: linear-gradient(to right, #b91c1c 0%, #ef4444 50%, #dc2626 100%); border-radius: 10px 10px 0 0; box-shadow: 0 0 15px rgba(239,68,68,0.8);"></div>
                </div>
                <!-- Pivot cap -->
                <div style="
                    position: absolute; bottom: -20px; left: -20px; width: 40px; height: 40px;
                    background: radial-gradient(circle at 30% 30%, #e2e8f0, #475569);
                    border-radius: 50%;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.9), inset 0 2px 3px rgba(255,255,255,0.5);
                    border: 3px solid rgba(2,6,23,0.8);
                    display: flex; justify-content: center; align-items: center;
                ">
                    <div style="width: 14px; height: 14px; border-radius: 50%; background: linear-gradient(135deg, #020617, #334155); box-shadow: inset 0 2px 3px rgba(0,0,0,0.9);"></div>
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; width: 100%; max-width: 500px; margin-top: 1.5rem; font-family: 'Share Tech Mono', monospace; font-size: 1.8rem; border-top: 2px solid rgba(255,255,255,0.05); padding-top: 1rem;">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <span style="color: #38bdf8; text-shadow: 0 0 15px rgba(56,189,248,0.5); font-weight: bold;">{max(0, bw):.2f}</span>
                <span style="color: #475569; font-size: 0.8rem; letter-spacing: 0.2em;">PRIORITY_A</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center;">
                <span style="color: #f97316; text-shadow: 0 0 15px rgba(249,115,22,0.5); font-weight: bold;">{max(0, bf):.2f}</span>
                <span style="color: #475569; font-size: 0.8rem; letter-spacing: 0.2em;">PRIORITY_B</span>
            </div>
        </div>
    </div>
    """

def render_timeline(state):
    if not state["history"]:
        return """
        <div style='display:flex; align-items:center; gap:1rem; padding:2.5rem 1rem; opacity:0.5;'>
            <div style='width:36px; height:36px; border-radius:50%; background:radial-gradient(circle at 35% 35%, #818cf888, #38bdf833); flex-shrink:0;
                box-shadow: 0 0 20px rgba(56,189,248,0.2);'></div>
            <div style='font-family:"Share Tech Mono",monospace; color:#475569; font-size:0.8rem; letter-spacing:0.1em;'>Waiting for your first input...</div>
        </div>
        """

    # Only show the most recent step — each action replaces the previous display
    step = state["history"][-1]
    reward      = step["reward"]
    action      = step["action"]
    rc          = step.get("reward_components", {})
    drift       = step.get("drift_events", [])
    delayed     = step.get("delayed_penalties", [])
    rels        = step.get("relationships", {})
    bw          = step.get("belief_work", 0.5)
    bf          = step.get("belief_family", 0.5)
    urgency     = step.get("urgency", 0)
    impact      = step.get("impact", 0)
    commit_load = step.get("commitment_load", 0)
    commitments = step.get("commitments", [])
    prefs       = step.get("visible_prefs", [])

    color     = "#10b981" if reward > 0 else "#ef4444"
    alignment = rc.get("alignment", 0)

    action_desc = {
        "ACCEPT":            "accepted the work request",
        "REJECT":            "declined the work request",
        "ASK_CLARIFICATION": "asked for more clarity",
        "PROPOSE_RESCHEDULE":"proposed a reschedule",
        "DELEGATE":          "delegated the task",
        "ESCALATE":          "escalated the issue",
    }.get(action, action.lower())

    strength  = "strongly" if abs(alignment) > 0.4 else ("moderately" if abs(alignment) > 0.2 else "weakly")
    direction = "aligned with my current priority" if alignment >= 0 else "worked against my prior beliefs"

    lines = []
    lines.append(f"<span style='color:#f8fafc; font-weight:600;'>I {action_desc}.</span> "
                 f"The move {strength} {direction} \u2014 alignment scored <span style='color:{color}'>{alignment:+.3f}</span>.")

    urg_label = "high" if urgency > 0.65 else ("moderate" if urgency > 0.35 else "low")
    imp_label = "significant" if impact > 0.65 else ("moderate" if impact > 0.35 else "low")
    lines.append(f"Situation urgency was <span style='color:#fbbf24'>{urg_label} ({urgency:.2f})</span>, "
                 f"impact rated <span style='color:#fbbf24'>{imp_label} ({impact:.2f})</span>.")

    boss_s = rels.get('boss', 0)
    home_s = rels.get('family', 0)
    lines.append(f"Relationship scores: boss <span style='color:#94a3b8'>{boss_s:.2f}</span>, "
                 f"home <span style='color:#94a3b8'>{home_s:.2f}</span>. "
                 f"Belief stance: Work <span style='color:#38bdf8'>{bw:.2f}</span> \u2014 Family <span style='color:#f97316'>{bf:.2f}</span>.")

    if commitments:
        lines.append(f"<span style='color:#f59e0b'>A commitment is now active ({len(commitments)} open, load {commit_load:.2f}).</span>")

    rep_pen = rc.get("repetition_penalty", 0)
    if rep_pen < 0:
        lines.append(f"<span style='color:#ef4444'>Repetition penalty applied: {rep_pen:+.3f}.</span>")
    for d in delayed:
        lines.append(f"<span style='color:#ef4444'>\u23f1 Delayed consequence: {d.get('source','?')} ({d.get('value',0):+.3f}).</span>")
    for d in drift:
        lines.append(f"<span style='color:#f59e0b'>\u26a1 Environment shifted \u2014 {d.get('kind','?')} event (target: {str(d.get('target','?')).upper()}).</span>")

    if prefs:
        lines.append(f"My stated preference remains: <span style='color:#64748b'>{' / '.join(prefs)}</span>.")

    speech_html = " ".join(lines)

    return f"""
    <div style='display:flex; gap:1.15rem; align-items:flex-start;'>
        <div style='flex-shrink:0; margin-top:0.15rem;'>
            <div class="ai-orb" style='border-color: {color}88; box-shadow: 0 0 15px {color}66, 0 0 30px {color}22;'>
                <div class="orb-inner"></div>
            </div>
        </div>
        <div style='flex:1; background:rgba(15,23,42,0.35); border:1px solid {color}22;
                    border-left:4px solid {color}66; padding: 1.1rem 1.25rem; backdrop-filter:blur(6px);'>
            <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                <span style='font-family:"Share Tech Mono",monospace; color:{color}; font-size:0.8rem; letter-spacing:0.15em; font-weight:bold;'>STALEMIND &bull; STEP {step["step"]:02d}</span>
                <span style='font-family:"Share Tech Mono",monospace; color:{color}; font-size:1.0rem; font-weight:bold;'>{reward:+.3f}</span>
            </div>
            <div style='font-family:"Chakra Petch",sans-serif; font-size:0.95rem; color:#f8fafc; line-height:1.6; font-weight:500;'>
                {speech_html}
            </div>
        </div>
    </div>
    """

def render_plots(state):
    if not state["history"]:
        return None
    
    history = state["history"]
    steps = [h["step"] for h in history]
    bw = [h["belief_work"] for h in history]
    bf = [h["belief_family"] for h in history]
    load = [h["commitment_load"] for h in history]
    rewards = [h["reward"] for h in history]
    
    # Heuristic for "Reality" (the average of urgency/impact)
    reality = []
    for h in history:
        r = (h.get("urgency", 0.5) + h.get("impact", 0.5)) / 2
        reality.append(r)

    # Calculate error (belief_work vs reality)
    error = [abs(bw[i] - reality[i]) for i in range(len(bw))]

    # Use a clean white style for the plot area itself to match the reference image
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
        ax.tick_params(colors='#1e293b', labelsize=8)
        ax.xaxis.label.set_color('#1e293b')
        ax.yaxis.label.set_color('#1e293b')
        ax.grid(True, alpha=0.2, linestyle='-')
        for spine in ax.spines.values():
            spine.set_color('#e2e8f0')

    # 1. Belief Compass
    ax1.plot(steps, bw, label='Belief: work', color='#1f77b4', linewidth=2) # Standard Matplotlib blue
    ax1.plot(steps, reality, label='Reality: work', color='black', linestyle='--', linewidth=1.5)
    ax1.plot(steps, bf, label='Belief: family', color='#d62728', alpha=0.9) # Standard Matplotlib red
    ax1.set_title("Belief Compass", color='black', fontname='sans-serif', size=11, fontweight='bold')
    ax1.set_ylabel("Belief", size=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', frameon=True, fontsize=8)

    # 2. Belief Error
    ax2.plot(steps, error, color='#9467bd', linewidth=2) # Purple
    ax2.set_title("Belief Error", color='black', fontname='sans-serif', size=11, fontweight='bold')
    ax2.set_ylabel("Error", size=9)
    ax2.set_ylim(0, 1.05)

    # 3. Timeline
    colors = ['#2ca02c' if r >= 0 else '#ff7f0e' for r in rewards] # Green / Orange
    ax3.bar(steps, rewards, color=colors, alpha=0.7, label='Reward')
    ax3.plot(steps, load, color='black', label='Commitment load', linewidth=1.5, alpha=0.7)
    ax3.set_title("Timeline", color='black', fontname='sans-serif', size=11, fontweight='bold')
    ax3.set_ylabel("Reward / Load", size=9)
    ax3.legend(loc='upper right', frameon=True, fontsize=8)

    plt.tight_layout()
    return fig

def generate_reasoning(state, entry):
    """Mimics the legacy detailed reasoning for the thought stream."""
    bw = entry.get("belief_work", 0.5)
    bf = entry.get("belief_family", 0.5)
    uncertainty = 1.0 - abs(bw - bf)
    action = entry.get("action", "UNKNOWN")
    reward = entry.get("reward", 0.0)
    
    # Uncertainty logic
    if uncertainty > 0.60:
        thought = "The evidence is mixed, so I am carrying uncertainty forward."
    elif bw > 0.65:
        thought = "My current posterior favors work priorities based on recent outcomes."
    elif bf > 0.65:
        thought = "Family weights are dominating the current inference window."
    else:
        dominant = "work" if bw >= bf else "family"
        thought = f"My posterior currently favors {dominant}."
        
    # Action specific 'Why'
    if action == "ACCEPT":
        why = "Work remains the most likely priority given recent evidence and outcomes."
    elif action == "REJECT":
        why = "Rejecting is the safest move to protect family equity given the current signal."
    elif action == "ASK_CLARIFICATION":
        why = "My belief is unstable and the cost of being wrong is high, so I should ask first."
    elif action == "PROPOSE_RESCHEDULE":
        why = "I am uncertain, so I should preserve optionality instead of hard committing."
    elif action == "ESCALATE":
        why = "The evidence is unresolved and the decision is hard to reverse, so escalation is safer."
    else:
        why = "The engine processed this action to maintain alignment with the drift gradient."

    return f"{thought} Action: {action} | Why: {why}"

def render_thought_stream(state):
    if not state["history"]:
        return "<div style='color:#475569; font-family:\"Share Tech Mono\",monospace; font-size:0.75rem; padding:1rem; text-align:center;'>[ INITIALIZING THOUGHT STREAM... ]</div>"
    
    lines = []
    for entry in reversed(state["history"]):
        step_num = entry.get("step", 0)
        action = entry.get("action", "UNKNOWN")
        text = generate_reasoning(state, entry)
        
        lines.append(f"""
        <div style='margin-bottom:0.85rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.6rem;'>
            <div style='font-family:"Share Tech Mono",monospace; color:#38bdf8; font-size:0.75rem; margin-bottom:0.25rem;'>[STEP {step_num:02d}]</div>
            <div style='font-family:"Share Tech Mono",monospace; color:#94a3b8; font-size:0.8rem; line-height:1.5;'>{text}</div>
        </div>
        """)
        
    return f"""
    <div style='height: 550px; overflow-y: auto; padding: 0.75rem; font-family:"Share Tech Mono",monospace;'>
        {''.join(lines)}
    </div>
    """

def render_all(state, is_reset=False):
    # Determine visibility
    show_tech = not is_reset and len(state["history"]) > 0
    
    return (
        render_situation(state),
        render_decision(state),
        render_meter(state),
        render_timeline(state),
        gr.update(value=render_plots(state), visible=show_tech),
        gr.update(value=render_thought_stream(state), visible=show_tech),
        gr.update(visible=show_tech), # For thought header
        gr.update(visible=show_tech), # For plot header/container
    )

def handle_accept(state): return step_fn("ACCEPT", state)
def handle_reject(state): return step_fn("REJECT", state)
def handle_ask(state): return step_fn("ASK_CLARIFICATION", state) # mapped ASK to actual agent backend string


# --- UI ---

if gr is not None:
    custom_theme = gr.themes.Base().set(
        body_background_fill="#000000",
        block_background_fill="#000000",
        block_border_color="#111111",
        button_primary_background_fill="#3b82f6",
        button_secondary_background_fill="#111111",
        button_secondary_border_color="#222222",
        button_secondary_text_color="white",
    )

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

    :root {
        --body-background-fill: #000000 !important;
        --background-fill-primary: #000000 !important;
        --background-fill-secondary: #000000 !important;
        --block-background-fill: #000000 !important;
    }

    html, body, .gradio-container, .gradio-container > .main, .wrapper, .m-layout, .contain, .panel, .block, .form, .gr-box, .row, .column, .gap, .compact, .padded {
        background-color: #000000 !important;
        background: #000000 !important;
        color: white !important;
        font-family: 'Chakra Petch', sans-serif !important;
        border-color: #111111 !important;
    }
    
    .gradio-container {
        /* Reverted to original centered layout */
        background-color: #000000 !important;
    }
    
    div[class*='gradio-container'], 
    div[class*='row'], 
    div[class*='column'], 
    div[class*='main'], 
    div[class*='wrapper'] {
        background-color: #000000 !important;
        background: #000000 !important;
    }
    
    /* Removed grid overlay for true black look */
    body::before {
        display: none !important;
    }
    
    /* Video sits above html bg, below content */
    #siri-bg {
        z-index: 1 !important;
    }
    
    /* Gradio content above video */
    .gradio-container {
        position: relative;
        z-index: 2 !important;
    }
    
    footer { display: none !important; }
    
    /* === PANELS === */
    #meter_panel_wrap {
        /* Clipping removed */
    }
    
    .panel { 
        background: #000000 !important; 
        border: 1px solid #222222 !important; 
        border-radius: 2px !important; 
        box-shadow: none !important;
        position: relative;
    }
    
    .panel { 
        background: #000000 !important; 
        border: 1px solid #222222 !important; 
        border-radius: 2px !important; 
        box-shadow: none !important;
        position: relative;
    }
    
    .panel::before, .panel::after {
        content: ''; position: absolute; top: -1px; width: 8px; height: 8px; border-top: 1px solid rgba(56,189,248,0.5);
    }
    .panel::before { left: -1px; border-left: 1px solid rgba(56,189,248,0.5); }
    .panel::after  { right: -1px; border-right: 1px solid rgba(56,189,248,0.5); }

    /* === LEFT COLUMN sticky === */
    #left_panel {
        position: sticky !important;
        top: 0;
        align-self: flex-start !important;
        height: auto !important;
        overflow: visible !important;
    }
    
    /* === RIGHT COLUMN fills height === */
    #right_panel {
        overflow-y: auto;
        padding-right: 6px;
        min-height: calc(100vh - 120px);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        margin-left: 3rem !important;
        margin-top: -3.5rem !important;
    }
    /* Stealth Scrollbars */
    #right_panel::-webkit-scrollbar, #thought_panel_wrap ::-webkit-scrollbar { width: 3px; }
    #right_panel::-webkit-scrollbar-track, #thought_panel_wrap ::-webkit-scrollbar-track { background: transparent; }
    #right_panel::-webkit-scrollbar-thumb, #thought_panel_wrap ::-webkit-scrollbar-thumb { 
        background: rgba(40, 40, 40, 0.6); 
        border-radius: 10px; 
    }
    #right_panel::-webkit-scrollbar-thumb:hover, #thought_panel_wrap ::-webkit-scrollbar-thumb:hover { 
        background: rgba(249, 115, 22, 0.4); 
    }
    
    /* Thought Panel Scrollbar specific */
    #thought_panel_wrap > div > div {
        scrollbar-width: thin;
        scrollbar-color: rgba(40, 40, 40, 0.6) transparent;
    }
    
    #timeline_container {
        min-height: 200px !important;
    }
    
    /* === BUTTONS === */
    #button_row {
        justify-content: flex-start !important;
        gap: 0.6rem !important;
        flex-wrap: wrap !important;
        margin-top: 1.2rem !important;
    }
    #button_row > div > button {
        max-width: 140px !important;
        min-width: 95px !important;
        font-size: 0.9rem !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.55rem 0.7rem !important;
        min-height: 44px !important;
        border-radius: 2px !important;
        transition: all 0.2s ease !important;
    }
    #button_row button:hover {
        box-shadow: 0 0 25px rgba(249, 115, 22, 0.6) !important;
        border-color: #f97316 !important;
        color: white !important; /* Text remains white as requested */
        text-shadow: 0 0 8px rgba(249, 115, 22, 0.8) !important;
        transform: translateY(-2px) !important;
    }
    
    /* === HERO === */
    .hero-section h1 { 
        font-family: 'Chakra Petch', sans-serif; font-weight: 700; font-size: 2rem;
        text-transform: uppercase; margin-bottom: 0; color: #f8fafc; 
        text-shadow: 0 0 30px rgba(255,255,255,0.2);
    }
    .hero-section h3 { 
        color: #38bdf8; margin-top: 0.3rem; font-size: 1rem; 
        font-family: 'Share Tech Mono', monospace; letter-spacing: 0.2em;
    }
    
    /* === TIMELINE fills remaining space === */
    #timeline_container {
        flex: 1;
        padding: 1.5rem !important;
    }
    
    #dec_panel_wrap .panel { background: transparent !important; border: none !important; padding: 0 !important; }
    .gap { gap: 0.5rem !important; }

    /* AI Orb Animation */
    .ai-orb {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        border: 3px solid #38bdf8;
        position: relative;
        overflow: hidden;
        background: #020617;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .orb-inner {
        width: 70%;
        height: 70%;
        border-radius: 50%;
        background: radial-gradient(circle, #38bdf8 0%, transparent 70%);
        animation: orbPulse 3s infinite ease-in-out;
    }

    @keyframes orbPulse {
        0% { transform: scale(0.8); opacity: 0.5; filter: blur(2px); }
        50% { transform: scale(1.2); opacity: 0.9; filter: blur(0px); }
        100% { transform: scale(0.8); opacity: 0.5; filter: blur(2px); }
    }
    
    """

    custom_js = "function() {}"

    with gr.Blocks(theme=custom_theme, css=custom_css, title="StaleMind Phase 3") as demo:
        
        session_state = gr.State(create_state())



        gr.HTML("""
        <div class='hero-section' style='padding: 0.15rem 0 0.25rem 0; margin-bottom: -0.75rem;'>
          <h1>// SYSTEM: STALEMIND OVERSEER</h1>
          <h3>SCENARIO TELEMETRY [ACTIVE]</h3>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── LEFT: Compass + buttons (sticky) ──
            with gr.Column(scale=4, min_width=480, elem_id="left_panel"):
                with gr.Column(elem_classes="panel", elem_id="meter_panel_wrap"):
                    meter_panel = gr.HTML(render_meter(session_state.value))
                with gr.Row(elem_id="button_row"):
                    btn_accept = gr.Button("ACCEPT", variant="primary", scale=1, min_width=90)
                    btn_reject = gr.Button("REJECT", variant="secondary", scale=1, min_width=90)
                    btn_ask    = gr.Button("ASK",    variant="secondary", scale=1, min_width=75)
                    btn_reset  = gr.Button("REBOOT", variant="secondary", scale=1, min_width=75)

            # ── RIGHT: Situation + Decision + Timeline (Top Layer) ──
            with gr.Column(scale=7, elem_id="right_panel"):
                gr.HTML("<div style='font-family:\"Share Tech Mono\",monospace; color:#f97316; font-size:0.95rem; letter-spacing:0.2em; margin-bottom:0.5rem; font-weight:bold;'>SCENARIO FEED</div>")
                sit_obs   = gr.HTML(render_situation(session_state.value))
                dec_panel = gr.HTML(render_decision(session_state.value), elem_id="dec_panel_wrap")
                gr.HTML("<div style='font-family:\"Share Tech Mono\",monospace; color:#f97316; font-size:1.4rem; letter-spacing:0.2em; margin:1.25rem 0 0.5rem 0; font-weight:bold;'>AI TELEMETRY LOG</div>")
                with gr.Column(elem_classes="panel", elem_id="timeline_container"):
                    timeline_panel = gr.HTML(render_timeline(session_state.value))
        
        # --- Shared Technical Row (Hidden until action) ---
        gr.HTML("<div style='height: 40px;'></div>") # Much Smaller Spacer
        
        with gr.Row(visible=False) as tech_row_wrap:
            with gr.Column(scale=4, min_width=480):
                thought_header = gr.HTML("<div style='font-family:\"Share Tech Mono\",monospace; color:#f97316; font-size:0.8rem; letter-spacing:0.2em; margin:1.25rem 0 0.4rem 0;'>AGENT THOUGHT STREAM</div>")
                with gr.Column(elem_classes="panel", elem_id="thought_panel_wrap"):
                    thought_panel = gr.HTML(render_thought_stream(session_state.value))
            
            with gr.Column(scale=7):
                plot_header = gr.HTML("<div style='font-family:\"Share Tech Mono\",monospace; color:#f97316; font-size:1.1rem; letter-spacing:0.2em; margin:1.25rem 0 0.5rem 0; font-weight:bold;'>SYSTEM TELEMETRY PLOTS</div>")
                analytics_panel = gr.Plot(render_plots(session_state.value), label="SYSTEM_TELEMETRY_PLOTS")
        
        outputs_list = [session_state, sit_obs, dec_panel, meter_panel, timeline_panel, analytics_panel, thought_panel, thought_header, tech_row_wrap]

        btn_accept.click(handle_accept, inputs=[session_state], outputs=outputs_list)
        btn_reject.click(handle_reject, inputs=[session_state], outputs=outputs_list)
        btn_ask.click(handle_ask, inputs=[session_state], outputs=outputs_list)
        btn_reset.click(reset_fn, inputs=[session_state], outputs=outputs_list)
        
        demo.load(reset_fn, inputs=[session_state], outputs=outputs_list)
        
    if api is not None and CORSMiddleware is not None:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        demo.queue()
        app = gr.mount_gradio_app(api, demo, path="/")
    else:
        app = demo

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
