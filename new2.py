import numpy as np
import random

# Define action space (kicker's choices within [0, 20])
num_actions = 21  # Represents discrete kicking locations from 0 to 20

def compute_reward(x_k, p_a, p_b, p_c):
    """Computes the reward based on the kicker's action."""
    if x_k < 5 or x_k > 15:
        return -5  # Constant penalty for missing the goal
    
    # Compute the reward function
    numerator = (p_a**2 / p_b) * (p_b**2 / (p_a * p_c))**(np.log(2/5) / np.log(4/3))
    denominator = ((p_a / p_b) * (p_b**2 / (p_a * p_c))**(np.log(2) / np.log(4/3)))**(x_k / 5)
    multiplier = x_k**(np.log(p_b**2 / (p_a * p_c)) / np.log(4/3) )
    reward = 1 - (numerator / denominator) * multiplier
    
    return reward

# Q-learning parameters
alpha_init = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon_init = 0.2  # Exploration rate
num_episodes = 100000  # Training episodes per goalkeeper
inter_goalkeeper_decay_init = 0.99  # Retention decay factor

# Initialize Q-values
Q_values = np.zeros(num_actions)

# Generate 500 random goalkeeper ability vectors
goalkeeper_samples = np.random.rand(500, 3)

# Track cumulative kicker probabilities
cumulative_kicker_probs = np.zeros(num_actions)

# Training loop across multiple goalkeepers
for i, (p_a, p_b, p_c) in enumerate(goalkeeper_samples):
    print(f"\nTraining Goalkeeper {i + 1}/{len(goalkeeper_samples)}")

    # Reduce the learning rate and exploration rate as training progresses
    alpha = alpha_init / (1 + 0.01 * i)
    epsilon = epsilon_init / (1 + 0.01 * i)
    inter_goalkeeper_decay = inter_goalkeeper_decay_init * (0.995 ** i)

    # Apply decay to retain past learning while adapting
    Q_values *= inter_goalkeeper_decay

    # Training episodes for current goalkeeper
    for episode in range(num_episodes):
        if episode % 10000 == 0:
            print(f"  Episode {episode}/{num_episodes}")

        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)  # Explore
        else:
            action = np.argmax(Q_values)  # Exploit best known action
        
        # Convert action index to actual x_k value (0 to 20 range)
        x_k = action  # Ensuring the kicker chooses anywhere from [0,20]
        
        # Compute reward
        reward = compute_reward(x_k, p_a, p_b, p_c)
        
        # Q-value update
        Q_values[action] += alpha * (reward + gamma * np.max(Q_values) - Q_values[action])

    # Update kicker probabilities using softmax
    new_kicker_probs = np.exp(Q_values) / np.sum(np.exp(Q_values))
    cumulative_kicker_probs += new_kicker_probs
    
    print(f"Goalkeeper {i+1} Probabilities: {[round(p, 3) for p in [p_a, p_b, p_c]]}")
    print(f"Kicker {i+1} Probabilities: {[round(p, 3) for p in new_kicker_probs]}")

# Compute final cumulative probabilities
final_probs = {str(i): round(cumulative_kicker_probs[i] / len(goalkeeper_samples), 3) for i in range(num_actions)}
print("\nFinal Cumulative Kicker Probabilities:", final_probs)