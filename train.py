import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os
from cliff_walker import CliffWalker  # Import your custom environment class

# --- SETTINGS (Hyperparameters) ---
EPISODES = 500          # Total training episodes
LEARNING_RATE = 0.1     # (Alpha) Learning Rate: how quickly the agent updates old values
DISCOUNT_FACTOR = 0.99  # (Gamma) Discount Factor: importance of future rewards
EPSILON = 1.0           # (Exploration) Probability of random action
EPSILON_DECAY = 0.99    # How fast we reduce randomness
EPSILON_MIN = 0.01      # Minimum randomness (1%)

# 1. Initialize environment WITHOUT rendering (for speed)
env = CliffWalker(render_mode=None)

# Create Q-Table: [Number of States x Number of Actions]
# We have 48 cells (4x12) and 4 actions.
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# List to store reward history
rewards_history = []

print("🚀 Starting agent training (Q-Learning)...")

# --- MAIN TRAINING LOOP ---
for episode in range(EPISODES):
    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # A. Action Selection (Epsilon-Greedy)
        # If random number < epsilon -> Explore (random action)
        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        # Else -> Exploit (best known action)
        else:
            action = np.argmax(q_table[state])

        # B. Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)

        # C. Update Q-Table (Bellman Equation)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Q_new = Q_old + lr * (Reward + discount * max_future_q - Q_old)
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

    # Store episode result
    rewards_history.append(total_reward)

    # Decay epsilon (agent becomes less chaotic)
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Print progress every 50 episodes
    if episode % 50 == 0:
        print(f"Episode {episode}: Reward {total_reward:.0f}, Epsilon {EPSILON:.2f}")

print("\n✅ Training finished!")
env.close()

# --- RESULT DEMONSTRATION ---
print("🎥 Starting demo with rendering...")
time.sleep(1)

# Create new environment WITH rendering
env = CliffWalker(render_mode="human")
state, _ = env.reset()
done = False

while not done:
    # Now choose ONLY the best actions (argmax)
    action = np.argmax(q_table[state])
    
    state, reward, terminated, truncated, _ = env.step(action)
    
    # Small pause to see the movement
    # (although CliffWalker class has FPS limit, this is just in case)
    # time.sleep(0.1) 
    
    if terminated:
        print("🏆 VICTORY! Agent reached the goal!")
        done = True
        time.sleep(2) # Delay before closing
    elif truncated:
        print("❌ Time out.")
        done = True

env.close()

# --- PLOTTING ---
print("📊 Plotting graph...")

# Creates the 'plots' folder if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.title('Agent Training Process')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)

# Saves the graph
plt.savefig('plots/training_result.png')
print("✅ Graph saved to plots/training_result.png")

plt.show() # Opens the graph window