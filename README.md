# 🧊 Cliff Walker RL Agent (Winter School 2026)

This project demonstrates a **Reinforcement Learning (RL)** agent trained to solve the classic "Cliff Walker" problem using **Q-Learning**.

The code was generated using a "Vibe Coding" approach with **Local LLMs (LM Studio)** and refined for optimal performance.

---

## The "Vibe Coding" Workflow

This project was built by interacting with a local Large Language Model (`Qwen 2.5 Coder 7B`) via LM Studio.

### 1. Model Selection
I used **Qwen 2.5 Coder 7B (Instruct)**, running locally. This model was chosen for its superior reasoning capabilities in Python and Algorithm generation.

![Model Selection](/docs/screenshots/Screenshot_20260204_225658-2.png)

### 2. Prompt Engineering
I used a specific **System Prompt** to enforce strict coding standards (no placeholders, full implementation) and a structured **User Prompt** to define the game physics (Grid 4x12, Cliff penalties, Rewards).

![Prompt Engineering](/docs/screenshots/Screenshot_20260204_225830.png)

![Prompt Engineering](/docs/screenshots/Screenshot_20260204_230028.png)

![Prompt Engineering](/docs/screenshots/Screenshot_20260204_230038.png)

### 3. Code Generation
The model generated the custom Gymnasium environment (`CliffWalker` class) based on the specifications.

![Code Generation](/docs/screenshots/Screenshot_20260204_230751.png)

![Code Generation](/docs/screenshots/Screenshot_20260204_230808.png)

---

## Theory: Q-Learning

The agent learns to navigate the grid by maintaining a **Q-Table** (State-Action values).
* **State:** The agent's position on the 4x12 grid (0-47).
* **Actions:** Up, Right, Down, Left.
* **Rewards:**
    * `-1` per step (encourages shortest path).
    * `-100` for falling into the Cliff (and reset to start).
    * `+100` for reaching the Goal.

**Algorithm:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s',a') - Q(s,a)]$$

---

## Installation & Usage

### Prerequisites
* Python 3.10+
* Dependencies: `gymnasium`, `numpy`, `pygame`, `matplotlib`

### 1. Install Dependencies
```bash
pip install gymnasium numpy pygame matplotlib
```

### 2. Run Training
```bash
python train.py
```

---

## Results

### Training Process (Visual)

During the first 50 episodes, the agent explores randomly (High Epsilon). By episode 350, it converges to the optimal path, hugging the cliff edge without falling.

![Code Generation](/docs/screenshots/Screenshot_20260204_231921.png)

### Learning Curve

The graph below shows the total reward per episode.

* Early Episodes: Deep drops to -50000 indicate frequent falls into the cliff.

* Convergence: Around episode 100-150, the reward stabilizes near -13 (the optimal number of steps to reach the goal), proving the agent has mastered the game.

![Code Generation](/docs/screenshots/Screenshot_20260204_231947.png)

---

## Project Structure

```
├── docs/
│   └── screenshots/        # Images used in this README
├── plots/
│   └── training_result.png # Saved learning curve graph
├── cliff_walker.py         # Custom Gymnasium Environment logic
├── train.py                # Q-Learning algorithm & Training loop
├── requirements.txt        # List of dependencies
└── readme.md               # Project documentation
```
