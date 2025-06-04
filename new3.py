import numpy as np
import random

# Define action space (kicker's choices within [-5, 15] for both x and y)
x_range = np.arange(-5, 16)  # x values from -5 to 15
y_range = np.arange(-5, 16)  # y values from -5 to 15

# Define the goalkeeper's probability distribution in four quadrants
def compute_reward(x, y, p_a, p_b, p_c, p_d):
    """Computes the reward based on the kicker's shot location."""
    if x < 0 or x > 10 or y < 0 or y > 10:
        return -5  # Constant penalty for missing the goal
    
    # Compute the reward function
    numerator = (p_a**(9/8)) * (p_b**(((x - y) * (3*x - y)) / 100)) * (p_c**(((x - y) * (x - 3*y)) / 100)) * (p_a**(x*y / 25)) * (p_d**(x*y / 25))
    denominator = (p_d**(1/8)) * (p_a**((3/100) * (x**2 + y**2))) * (p_d**((1/100) * (x**2 + y**2)))
    reward = 1 - (numerator / denominator)
    
    return reward

# Q-learning parameters
alpha_init = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon_init = 0.2  # Exploration rate
num_episodes = 100000  # Training episodes per goalkeeper
inter_goalkeeper_decay_init = 0.99  # Retention decay factor

# Initialize Q-values
Q_values = np.zeros((len(x_range), len(y_range)))

# ⚽️ Train for only 100 footballers
num_footballers = 100  
goalkeeper_samples = np.random.rand(num_footballers, 4)  # pa, pb, pc, pd

# Track cumulative kicker probabilities
cumulative_kicker_probs = np.zeros((len(x_range), len(y_range)))

# Training loop across multiple goalkeepers
for i, (p_a, p_b, p_c, p_d) in enumerate(goalkeeper_samples):
    print(f"\nTraining Goalkeeper {i + 1}/{num_footballers}")

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
            x_idx = random.randint(0, len(x_range) - 1)  # Explore x
            y_idx = random.randint(0, len(y_range) - 1)  # Explore y
        else:
            x_idx, y_idx = np.unravel_index(np.argmax(Q_values), Q_values.shape)  # Exploit best known action
        
        x_k, y_k = x_range[x_idx], y_range[y_idx]  # Convert index to actual values
        
        # Compute reward
        reward = compute_reward(x_k, y_k, p_a, p_b, p_c, p_d)
        
        # Q-value update
        Q_values[x_idx, y_idx] += alpha * (reward + gamma * np.max(Q_values) - Q_values[x_idx, y_idx])

    # Update kicker probabilities using softmax
    exp_Q = np.exp(Q_values - np.max(Q_values))  # Numerical stability trick
    new_kicker_probs = exp_Q / np.sum(exp_Q)
    cumulative_kicker_probs += new_kicker_probs
    
    print(f"Goalkeeper {i+1} Probabilities: {[round(p, 3) for p in [p_a, p_b, p_c, p_d]]}")
    print("Kicker Probabilities:")
    for x_idx, x in enumerate(x_range):
        row_probs = [round(new_kicker_probs[x_idx, y_idx], 3) for y_idx, y in enumerate(y_range)]
        print(f"x={x}: {row_probs}")
    # Sum the full unrounded grid
    total_sum_precise = np.sum(new_kicker_probs)
    print(f"Precise sum of all kicker probabilities: {total_sum_precise}")


# Compute final cumulative probabilities
final_probs = {f"({x}, {y})": round(cumulative_kicker_probs[i, j] / num_footballers, 3) 
               for i, x in enumerate(x_range) for j, y in enumerate(y_range)}

print("\nFinal Cumulative Kicker Probabilities:")
for x_idx, x in enumerate(x_range):
    row_probs = [round(cumulative_kicker_probs[x_idx, y_idx] / num_footballers, 3) 
                 for y_idx, y in enumerate(y_range)]
    print(f"x={x}: {row_probs}")
