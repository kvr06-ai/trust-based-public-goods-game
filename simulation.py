import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Simulation Parameters
n_agents = 100
n_rounds = 500
endowment = 10
contribution_choices = [0, 2, 4, 6, 8, 10]
r = 1.5  # Public good multiplication factor
lambda_trust = 0.8  # Trust memory parameter
eta = 0.5  # Trust weight
alpha = 0.5  # Fairness parameter (disadvantageous inequity)
beta = 0.2  # Fairness parameter (advantageous inequity)
gamma = 0.3  # Social pressure weight
rlhf_frequency = 50  # RLHF every 50 rounds

# Q-learning parameters
alpha_q = 0.1  # Learning rate
gamma_q = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize agents
class Agent:
    def __init__(self):
        self.trust = 0.5  # Initial trust level
        self.q_table = np.zeros(len(contribution_choices))  # Q-values for each action
        self.contribution_history = []

# Utility functions
def rational_utility(c_i, total_contributions):
    return endowment - c_i + (r / n_agents) * total_contributions

def trust_utility(c_i, trust):
    return eta * trust * c_i

def fairness_utility(p_i, p_mean):
    return -alpha * max(p_mean - p_i, 0) - beta * max(p_i - p_mean, 0)

def social_pressure_utility(c_i, c_mean):
    return gamma * (1 - abs(c_i - c_mean))

def behavioral_utility(c_i, total_contributions, trust, p_i, p_mean, c_mean):
    u_rat = rational_utility(c_i, total_contributions)
    u_trust = trust_utility(c_i, trust)
    u_fair = fairness_utility(p_i, p_mean)
    u_soc = social_pressure_utility(c_i, c_mean)
    return u_rat + u_trust + u_fair + u_soc

# Simulation
agents = [Agent() for _ in range(n_agents)]
contributions_over_time = []
trust_over_time = []
payoff_variance_over_time = []

for t in range(n_rounds):
    # Agents choose contributions based on Q-learning
    contributions = []
    for agent in agents:
        if np.random.rand() < epsilon:
            # Exploration: random choice
            c_i = np.random.choice(contribution_choices)
        else:
            # Exploitation: choose best action
            c_i = contribution_choices[np.argmax(agent.q_table)]
        contributions.append(c_i)
        agent.contribution_history.append(c_i)
    
    total_contributions = sum(contributions)
    c_mean = np.mean(contributions)
    
    # Calculate rational payoffs
    payoffs = [rational_utility(c_i, total_contributions) for c_i in contributions]
    p_mean = np.mean(payoffs)
    
    # Update trust and Q-tables
    for i, agent in enumerate(agents):
        # Update trust
        group_avg_contribution = total_contributions / n_agents
        agent.trust = lambda_trust * agent.trust + (1 - lambda_trust) * (group_avg_contribution / max(contribution_choices))
        
        # Calculate behavioral utility
        u_beh = behavioral_utility(contributions[i], total_contributions, agent.trust, payoffs[i], p_mean, c_mean)
        
        # Update Q-table
        action_idx = contribution_choices.index(contributions[i])
        max_future_q = np.max(agent.q_table)
        agent.q_table[action_idx] = (1 - alpha_q) * agent.q_table[action_idx] + alpha_q * (u_beh + gamma_q * max_future_q)
    
    # RLHF: Adjust Q-tables every 50 rounds
    if (t + 1) % rlhf_frequency == 0:
        for agent in agents:
            # Simulate human feedback by increasing Q-values for higher contributions
            for idx, c in enumerate(contribution_choices):
                agent.q_table[idx] += 0.1 * c  # Bias towards higher contributions
    
    # Track metrics
    contributions_over_time.append(np.mean(contributions))
    trust_over_time.append(np.mean([agent.trust for agent in agents]))
    payoff_variance_over_time.append(np.var(payoffs))

# Save data to CSV
data = pd.DataFrame({
    'Round': range(n_rounds),
    'Average_Contribution': contributions_over_time,
    'Average_Trust': trust_over_time,
    'Payoff_Variance': payoff_variance_over_time
})
data.to_csv('simulation_results.csv', index=False)

# Plot results
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(contributions_over_time, color='#2E86AB', linewidth=2)  # Subtle blue
plt.title('Average Contribution Over Time', pad=15)
plt.xlabel('Round')
plt.ylabel('Average Contribution')

plt.subplot(3, 1, 2)
plt.plot(trust_over_time, color='#7EA172', linewidth=2)  # Subtle green
plt.title('Average Trust Over Time', pad=15)
plt.xlabel('Round')
plt.ylabel('Average Trust')

plt.subplot(3, 1, 3)
plt.plot(payoff_variance_over_time, color='#A23B72', linewidth=2)  # Subtle purple-red
plt.title('Payoff Variance Over Time', pad=15)
plt.xlabel('Round')
plt.ylabel('Payoff Variance')

plt.tight_layout(pad=2.0)
plt.savefig('simulation_plots.png', dpi=300, bbox_inches='tight')
plt.show()