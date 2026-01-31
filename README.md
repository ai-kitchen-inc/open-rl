# Reinforcement Learning (RL) — An Intuition-First Tutorial

This tutorial is for product managers and university students who want to **understand and ship** RL ideas without being buried in math. We'll keep it practical, visual, and grounded in a tiny but real decision-making problem with **delayed reward** (you don't know if a choice was good until later).

---

## What you’ll build
You’ll build a **tiny delivery robot** that learns to reach a goal in a grid while avoiding traps. You'll first try a baseline (random/heuristic), see why it fails, then train a **Q-learning** agent that learns from trial and error.

---

## Who it’s for
- Product managers who want intuition for RL trade-offs
- Students who want a first project without deep math
- Engineers who want a minimal, readable baseline

---

## Prerequisites (minimal)
- Python 3.9+
- `numpy`
- `matplotlib` (for plots)

Install:
```bash
pip install numpy matplotlib
```

---

## How to run
```bash
cd tutorial
python train.py
python plot.py
```

---

## What success looks like
You should see:
- Training rewards that trend upward (less negative, more positive).
- A saved `learning_curve.png` and `policy.png` in `tutorial/outputs`.
- An average evaluation reward that beats random (often above -5 by episode 400).

Example output:
```json
{
  "train_rewards": [...],
  "eval_rewards": [...],
  "avg_eval_reward": 3.6
}
```

---

# Lesson Content

## 1) Hook story (non-RL framing)
Imagine you’re managing a **smart delivery robot** in a warehouse. The robot sees only the nearby hallway. It has to deliver a package quickly, avoid dangerous areas, and it **only gets praised when the delivery is complete**. If it hits a hazard, you hear about it later. This is a delayed feedback problem—perfect for RL.

---

## 2) The decision-making loop in plain English
Forget jargon for a moment. The loop looks like this:
- **You** (the learner) are trying things.
- **The world** reacts.
- You get **a score** that tells you if that was good.
- You repeat and adjust.

Translated:
- **Agent**: the learner making choices (our robot).
- **Environment**: the world it lives in (the grid).
- **Action**: a choice (move up/down/left/right).
- **Reward**: the score (step cost, goal bonus, trap penalty).

This cycle happens again and again. Over time, good choices become more likely.

---

## 3) A tiny environment (from scratch)
We’ll use a **5x5 grid**:
- Start: bottom-left
- Goal: top-right (**+10 reward**)
- Traps: two cells (**-10 reward**)
- Every step costs **-1** (so wandering hurts)
- Episode ends when you hit goal, trap, or 30 steps

This gives us **delayed reward**: the robot only “wins” at the end.

---

## 4) Baseline solution + why it fails
Two baselines:
1) **Random walking** — finds the goal rarely and wastes time.
2) **Greedy heuristic** — “always move toward the goal.” This ignores traps and can fail badly.

Why it fails:
It doesn’t **learn from outcomes**. It repeats mistakes.

---

## 5) The RL improvement (Q-learning)
Q-learning updates a table of “how good is this action from this state?”
It doesn’t need a model of the world—just experience.

We’ll do:
- Try an action
- Observe reward and next state
- Update a score for that (state, action) pair

---

## 6) Visualizations of learning
- **Reward curve**: Does performance improve?
- **Policy plot**: Which action does the agent prefer in each cell?

---

## 7) Common mistakes + debugging checklist
- **Rewards too sparse** → learning stalls.
- **Exploration too low** → agent gets stuck.
- **Episode too short** → no time to reach goal.
- **Wrong discount factor** → shortsighted or overly delayed decisions.

Checklist:
- Are rewards non-zero sometimes?
- Do actions cover the space?
- Does training improve evaluation reward?
- Are plots being saved?

---

## 8) When NOT to use RL
- When you can write a direct rule or solve it with optimization.
- When data is tiny and errors are costly.
- When rewards are unclear or delayed too far.
- When you can’t simulate or safely experiment.

---

## 9) Industry mapping (transfer to real systems)
Gridworld ≈ real systems where actions have delayed impact:
- **Inventory re-ordering**: early decisions affect later stock-outs.
- **Call center staffing**: choices now affect wait times later.
- **Product recommendations**: short-term clicks vs long-term retention.

In real systems, RL is valuable when you can **simulate**, **evaluate safely**, and **measure outcomes**.

---

## Optional Math Corner
Q-learning update (optional):
```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
```
Where:
- α = learning rate
- γ = discount factor

---

# Code (clean + commented)

Below are the exact scripts you’ll run.

## `environment.py`
```python
import numpy as np


class GridworldEnv:
    """
    A tiny gridworld with delayed reward.

    - The agent starts at the bottom-left corner.
    - The goal is the top-right corner.
    - Every step costs -1.
    - Reaching the goal gives +10 and ends the episode.
    - Falling into a trap gives -10 and ends the episode.
    """

    ACTIONS = ["up", "right", "down", "left"]
    ACTION_TO_DELTA = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, size=5, max_steps=30, seed=7):
        self.size = size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.start = (size - 1, 0)
        self.goal = (0, size - 1)
        self.traps = {(1, 3), (2, 2)}

        self.state = None
        self.steps = 0

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return len(self.ACTIONS)

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self._state_to_index(self.state)

    def step(self, action):
        self.steps += 1
        row, col = self.state
        dr, dc = self.ACTION_TO_DELTA[action]
        next_row = int(np.clip(row + dr, 0, self.size - 1))
        next_col = int(np.clip(col + dc, 0, self.size - 1))
        self.state = (next_row, next_col)

        reward = -1.0
        done = False

        if self.state == self.goal:
            reward = 10.0
            done = True
        elif self.state in self.traps:
            reward = -10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._state_to_index(self.state), reward, done, {}

    def _state_to_index(self, state):
        row, col = state
        return row * self.size + col

    def index_to_state(self, index):
        row = index // self.size
        col = index % self.size
        return (row, col)
```

## `agent.py`
```python
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.2,
        discount=0.95,
        epsilon=0.2,
        seed=7,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        target = reward + (0.0 if done else self.discount * best_next)
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def greedy_action(self, state):
        return int(np.argmax(self.q_table[state]))
```

## `train.py`
```python
import argparse
import json
from pathlib import Path

import numpy as np

from environment import GridworldEnv
from agent import QLearningAgent


def run_episode(env, agent, explore=True):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state) if explore else agent.greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        if explore:
            agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

    return total_reward


def train(args):
    env = GridworldEnv(size=5, max_steps=30, seed=args.seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    rewards = []
    for _ in range(args.episodes):
        episode_reward = run_episode(env, agent, explore=True)
        rewards.append(episode_reward)

    eval_rewards = [run_episode(env, agent, explore=False) for _ in range(20)]
    results = {
        "train_rewards": rewards,
        "eval_rewards": eval_rewards,
        "avg_eval_reward": float(np.mean(eval_rewards)),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(output_dir / "q_table.npy", agent.q_table)

    with (output_dir / "results.json").open("w") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
```

## `plot.py`
```python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment import GridworldEnv


def plot_learning_curve(rewards, output_path):
    window = 20
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        smooth = rewards

    plt.figure(figsize=(7, 4))
    plt.plot(rewards, alpha=0.4, label="episode reward")
    plt.plot(np.arange(len(smooth)) + (window - 1), smooth, label="moving avg")
    plt.title("Learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_policy(q_table, output_path):
    env = GridworldEnv()
    policy = np.argmax(q_table, axis=1).reshape(env.size, env.size)

    arrow_map = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
    }

    plt.figure(figsize=(4, 4))
    plt.imshow(np.zeros((env.size, env.size)), cmap="Greys", vmin=0, vmax=1)
    for row in range(env.size):
        for col in range(env.size):
            symbol = arrow_map[policy[row, col]]
            if (row, col) == env.goal:
                symbol = "★"
            if (row, col) in env.traps:
                symbol = "✖"
            plt.text(col, row, symbol, ha="center", va="center", fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.title("Greedy policy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    rewards = np.load(output_dir / "train_rewards.npy")
    q_table = np.load(output_dir / "q_table.npy")

    plot_learning_curve(rewards, output_dir / "learning_curve.png")
    plot_policy(q_table, output_dir / "policy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training outputs.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    main(parser.parse_args())
```

---

## Exercises (with hints)
1) **Make it harder.** Add more traps.  
   *Hint:* Add coordinates to `self.traps`.

2) **Change the reward.** Give +1 for moving closer to the goal.  
   *Hint:* Compare Manhattan distance before and after each step.

3) **Tune exploration.** Try epsilon = 0.05 or 0.5.  
   *Expected:* Lower epsilon learns slowly but converges; higher epsilon explores more but plateaus.

---

## Extras (optional)
- Try `gymnasium` for larger environments.
- Try `stable-baselines3` for deep RL baselines.

---

# 10-Part Open-Source RL Series (Standalone Folders)

## 1) Series table of contents
1. `part-01-bandits` — Exploration vs. exploitation
2. `part-02-mdps` — Delayed rewards & state
3. `part-03-q-learning` — Value learning
4. `part-04-policy-gradient` — Directly learning a policy
5. `part-05-actor-critic` — Value + policy together
6. `part-06-model-based` — Planning with a learned model
7. `part-07-offline-rl` — Learning from logged data
8. `part-08-exploration` — Efficient exploration
9. `part-09-multi-agent` — Games and coordination
10. `part-10-safety-deployment` — Safety, monitoring, rollout

---

## 2) Folder structure
```
series/
  part-01-bandits/
  part-02-mdps/
  part-03-q-learning/
  part-04-policy-gradient/
  part-05-actor-critic/
  part-06-model-based/
  part-07-offline-rl/
  part-08-exploration/
  part-09-multi-agent/
  part-10-safety-deployment/
```

---

## 3) Part-by-part overview

### Part 01 — Bandits
- **Learning objectives:** Explore vs. exploit, reward uncertainty.
- **Environment:** 3-arm Bernoulli bandit.
- **Algorithm:** Epsilon-greedy.
- **Key plot:** Average reward over time.
- **Industry mapping:** A/B testing ad creatives to maximize click-through.

### Part 02 — MDPs
- **Learning objectives:** States, transitions, delayed rewards.
- **Environment:** Two-room navigation with a door.
- **Algorithm:** Random policy + value evaluation (Monte Carlo).
- **Key plot:** Value of states.
- **Industry mapping:** Customer journeys with delayed conversion.

### Part 03 — Q-learning
- **Learning objectives:** Learning action values from experience.
- **Environment:** Gridworld with traps.
- **Algorithm:** Tabular Q-learning.
- **Key plot:** Reward curve + greedy policy.
- **Industry mapping:** Warehouse routing with delayed delivery bonuses.

### Part 04 — Policy Gradient
- **Learning objectives:** Learn a policy directly via reward feedback.
- **Environment:** Two-step decision chain (delayed reward).
- **Algorithm:** REINFORCE (with baseline).
- **Key plot:** Policy probability over time.
- **Industry mapping:** Adjusting pricing strategy based on long-term revenue.

### Part 05 — Actor-Critic
- **Learning objectives:** Combine value and policy learning.
- **Environment:** Cliff-walk lite.
- **Algorithm:** Advantage actor-critic (tabular).
- **Key plot:** Advantage estimates + reward curve.
- **Industry mapping:** Balancing short-term service levels vs. long-term churn.

### Part 06 — Model-Based RL
- **Learning objectives:** Learn a model, then plan.
- **Environment:** Simple deterministic maze.
- **Algorithm:** Dyna-Q (planning updates).
- **Key plot:** Learning speed with/without planning.
- **Industry mapping:** Simulating supply-chain decisions before deployment.

### Part 07 — Offline RL Intuition
- **Learning objectives:** Learning from logged data only.
- **Environment:** Logged bandit data.
- **Algorithm:** Fitted Q Evaluation (tabular).
- **Key plot:** Estimated policy value vs. behavior policy.
- **Industry mapping:** Learning from historical customer interactions.

### Part 08 — Exploration
- **Learning objectives:** Efficient exploration.
- **Environment:** Sparse-reward grid.
- **Algorithm:** UCB bonuses.
- **Key plot:** Visits heatmap.
- **Industry mapping:** Finding new market segments with minimal trials.

### Part 09 — Multi-Agent
- **Learning objectives:** Coordination and competition.
- **Environment:** Two-agent coordination game.
- **Algorithm:** Independent Q-learning.
- **Key plot:** Joint action frequencies.
- **Industry mapping:** Pricing competition between platforms.

### Part 10 — Safety & Deployment
- **Learning objectives:** Safety constraints, rollout strategy.
- **Environment:** Safe grid with unsafe zones.
- **Algorithm:** Constrained Q-learning (penalty-based).
- **Key plot:** Reward vs. safety violations.
- **Industry mapping:** Autonomous system rollout with guardrails.
