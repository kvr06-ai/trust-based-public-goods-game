import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation Parameters
n_agents = 100              # Number of agents
n_rounds = 500              # Number of rounds
endowment = 10              # Initial endowment per round
contribution_choices = list(range(0, 11))  # Possible contributions: [0, 1, 2, ..., 10]
r = 50                      # Public good multiplication factor
lambda_trust = 0.5          # Trust memory parameter
eta = 1.0                   # Trust weight in utility
alpha = 1.0                 # Fairness parameter (disadvantageous inequity)
beta = 0.5                  # Fairness parameter (advantageous inequity)
gamma = 0.6                 # Social pressure weight
rlhf_frequency = 25         # Apply RLHF every 25 rounds
rlhf_boost = 0.2            # RLHF boost factor for Q-values

# Q-learning Parameters
alpha_q = 0.1               # Learning rate for Q-learning
gamma_q = 0.9               # Discount factor for Q-learning
epsilon = 0.1               # Exploration rate for epsilon-greedy policy

# Agent Class Definition
class Agent:
    def __init__(self):
        self.trust = 0.2                    # Initial trust level
        self.q_table = np.zeros(len(contribution_choices))  # Q-values for each contribution
        self.contribution_history = []      # Track contributions over time

# Utility Functions
def rational_utility(c_i, total_contributions):
    """Calculate rational utility based on contribution and public good return."""
    public_return = (r * total_contributions) / n_agents
    return endowment - c_i + public_return

def trust_utility(c_i, trust):
    """Calculate utility from trust in others' contributions."""
    return eta * trust * c_i

def fairness_utility(p_i, p_mean):
    """Calculate utility penalty from inequity in payoffs."""
    return -alpha * max(p_mean - p_i, 0) - beta * max(p_i - p_mean, 0)

def social_pressure_utility(c_i, c_mean):
    """Calculate utility from aligning with average contribution."""
    return gamma * (1 - abs(c_i - c_mean) / max(contribution_choices))

def behavioral_utility(c_i, total_contributions, trust, p_i, p_mean, c_mean):
    """Combine rational and behavioral utilities."""
    u_rat = rational_utility(c_i, total_contributions)
    u_trust = trust_utility(c_i, trust)
    u_fair = fairness_utility(p_i, p_mean)
    u_soc = social_pressure_utility(c_i, c_mean)
    return u_rat + u_trust + u_fair + u_soc

# Initialize Agents
agents = [Agent() for _ in range(n_agents)]

# Data Tracking
contributions_over_time = []
trust_over_time = []
payoff_variance_over_time = []

# Run Simulation
for t in range(n_rounds):
    # Agents choose contributions using Q-learning
    contributions = []
    for agent in agents:
        if np.random.rand() < epsilon:
            # Exploration: random contribution
            c_i = np.random.choice(contribution_choices)
        else:
            # Exploitation: choose best action from Q-table
            c_i = contribution_choices[np.argmax(agent.q_table)]
        contributions.append(c_i)
        agent.contribution_history.append(c_i)
    
    # Calculate totals and averages
    total_contributions = sum(contributions)
    c_mean = np.mean(contributions)
    payoffs = [rational_utility(c_i, total_contributions) for c_i in contributions]
    p_mean = np.mean(payoffs)
    
    # Update agents' trust and Q-tables
    for i, agent in enumerate(agents):
        # Update trust based on group average contribution
        group_avg_contribution = total_contributions / n_agents
        agent.trust = lambda_trust * agent.trust + (1 - lambda_trust) * (group_avg_contribution / max(contribution_choices))
        
        # Calculate total utility with behavioral factors
        u_beh = behavioral_utility(contributions[i], total_contributions, agent.trust, payoffs[i], p_mean, c_mean)
        
        # Update Q-table using Q-learning
        action_idx = contribution_choices.index(contributions[i])
        max_future_q = np.max(agent.q_table)
        agent.q_table[action_idx] = (1 - alpha_q) * agent.q_table[action_idx] + alpha_q * (u_beh + gamma_q * max_future_q)
    
    # Apply RLHF periodically
    if (t + 1) % rlhf_frequency == 0:
        for agent in agents:
            for idx, c in enumerate(contribution_choices):
                agent.q_table[idx] += rlhf_boost * c  # Boost Q-values for higher contributions
    
    # Record metrics
    contributions_over_time.append(np.mean(contributions))
    trust_over_time.append(np.mean([agent.trust for agent in agents]))
    payoff_variance_over_time.append(np.var(payoffs))

# Save Results to CSV
data = pd.DataFrame({
    'Round': range(n_rounds),
    'Average_Contribution': contributions_over_time,
    'Average_Trust': trust_over_time,
    'Payoff_Variance': payoff_variance_over_time
})
data.to_csv('simulation_results_updated.csv', index=False)

# Plot Results
plt.figure(figsize=(12, 8))

# Average Contribution Plot
plt.subplot(3, 1, 1)
plt.plot(contributions_over_time, color='blue')
plt.title('Average Contribution Over Time')
plt.xlabel('Round')
plt.ylabel('Average Contribution')

# Average Trust Plot
plt.subplot(3, 1, 2)
plt.plot(trust_over_time, color='green')
plt.title('Average Trust Over Time')
plt.xlabel('Round')
plt.ylabel('Average Trust')

# Payoff Variance Plot
plt.subplot(3, 1, 3)
plt.plot(payoff_variance_over_time, color='magenta')
plt.title('Payoff Variance Over Time')
plt.xlabel('Round')
plt.ylabel('Payoff Variance')

plt.tight_layout()
plt.savefig('simulation_plots_updated.png')
plt.show()