import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Simulation Parameters
n_agents = 100              # Number of agents
n_rounds = 500              # Number of rounds
endowment = 10              # Initial endowment per round
contribution_choices = np.arange(0, 10.5, 0.5)  # Contributions: 0 to 10 in steps of 0.5
r = 200                     # Public good multiplication factor (MPCR = 2.0)
lambda_trust = 0.4          # Trust memory parameter
eta = 2.0                   # Trust weight in utility
alpha = 4.0                 # Fairness: disadvantageous inequity aversion
beta = 3.0                  # Fairness: advantageous inequity aversion
gamma = 2.0                 # Social pressure weight
rlhf_frequency = 5          # Apply RLHF every 5 rounds
rlhf_boost = 0.8            # RLHF boost factor for Q-values
epsilon_initial = 0.2       # Initial exploration rate
epsilon_final = 0.01        # Final exploration rate
decay_rate = 0.01           # Decay rate for epsilon
punishment_strength = 0.3   # Strength of punishment mechanism
reward_cooperation = 0.5    # Additional reward for cooperation
rlhf_initial_boost = 2.0    # Strong initial boost to establish cooperation

# Q-learning Parameters
alpha_q = 0.2               # Learning rate for Q-learning
gamma_q = 0.95              # Discount factor for Q-learning

# Agent Class
class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.trust = 0.3                    # Higher initial trust level
        self.q_table = np.zeros(len(contribution_choices))  # Q-values for each contribution
        self.contribution_history = []      # Track contributions over time
        self.punishment_points = 0          # Points allocated to punish free-riders
        self.cooperation_streak = 0         # Track consecutive cooperative moves

    def is_cooperator(self):
        # Define a cooperator as someone who contributes above average
        if len(self.contribution_history) < 5:
            return False
        recent_contributions = self.contribution_history[-5:]
        return np.mean(recent_contributions) > 5.0

# Utility Functions
def rational_utility(c_i, total_contributions):
    public_return = (r * total_contributions) / n_agents
    return endowment - c_i + public_return

def trust_utility(c_i, trust):
    return eta * trust * c_i

def fairness_utility(p_i, p_mean, c_i, c_mean):
    # Enhanced fairness utility that considers both payoffs and contributions
    payoff_fairness = -alpha * max(p_mean - p_i, 0) - beta * max(p_i - p_mean, 0)
    contribution_fairness = -alpha * max(c_mean - c_i, 0) - beta * max(c_i - c_mean, 0)
    return (payoff_fairness + contribution_fairness) / 2

def social_pressure_utility(c_i, c_mean):
    return gamma * (1 - abs(c_i - c_mean) / max(contribution_choices))

def cooperation_reward(c_i, agent):
    # Reward consistent cooperation with bonus utility
    if c_i >= 5.0:  # Cooperative threshold
        agent.cooperation_streak += 1
    else:
        agent.cooperation_streak = 0
    
    # Reward for cooperation increases with streak, capped at 5
    streak_reward = min(agent.cooperation_streak, 5) * reward_cooperation
    return streak_reward if c_i >= 5.0 else 0

def punishment_utility(c_i, c_mean, agents, agent_idx):
    # Agents can punish those who contribute less than average
    punishment = 0
    
    if c_i < c_mean * 0.7:  # Free-riding threshold
        # Calculate punishment from other agents
        for other_agent in agents:
            if other_agent.id != agent_idx and other_agent.is_cooperator():
                # Cooperators punish free-riders
                punishment -= punishment_strength * (c_mean - c_i)
    
    return punishment

def behavioral_utility(c_i, total_contributions, trust, p_i, p_mean, c_mean, agents, agent_idx, agent):
    u_rat = rational_utility(c_i, total_contributions)
    u_trust = trust_utility(c_i, trust)
    u_fair = fairness_utility(p_i, p_mean, c_i, c_mean)
    u_soc = social_pressure_utility(c_i, c_mean)
    u_punish = punishment_utility(c_i, c_mean, agents, agent_idx)
    u_coop = cooperation_reward(c_i, agent)
    
    # Print detailed utility breakdown for a sample agent every 50 rounds
    # if agent_idx == 0 and len(agent.contribution_history) % 50 == 0:
    #     print(f"Agent 0 at round {len(agent.contribution_history)}:")
    #     print(f"  Rational: {u_rat:.2f}, Trust: {u_trust:.2f}, Fair: {u_fair:.2f}")
    #     print(f"  Social: {u_soc:.2f}, Punish: {u_punish:.2f}, Coop: {u_coop:.2f}")
    
    return u_rat + u_trust + u_fair + u_soc + u_punish + u_coop

# Initialize Agents with IDs
agents = [Agent(i) for i in range(n_agents)]

# Data Tracking
contributions_over_time = []
trust_over_time = []
payoff_variance_over_time = []
cooperation_rate_over_time = []

# Apply initial RLHF boost to establish cooperation
for agent in agents:
    for idx, c in enumerate(contribution_choices):
        # Strongly favor higher contributions initially
        agent.q_table[idx] += rlhf_initial_boost * (c / max(contribution_choices))**2

# Run Simulation
print("Starting simulation...")
for t in range(n_rounds):
    if t % 50 == 0:
        print(f"Running round {t}...")
    
    # Decaying epsilon for exploration-exploitation balance
    epsilon = epsilon_final + (epsilon_initial - epsilon_final) * np.exp(-decay_rate * t)
    
    # Agents choose contributions using Q-learning
    contributions = []
    for agent in agents:
        if np.random.rand() < epsilon:
            c_i = np.random.choice(contribution_choices)
        else:
            c_i = contribution_choices[np.argmax(agent.q_table)]
        contributions.append(c_i)
        agent.contribution_history.append(c_i)
    
    # Calculate totals and averages
    total_contributions = sum(contributions)
    c_mean = np.mean(contributions)
    payoffs = [rational_utility(c_i, total_contributions) for c_i in contributions]
    p_mean = np.mean(payoffs)
    
    # Calculate cooperation rate
    cooperators = sum(1 for c in contributions if c >= 5.0)
    cooperation_rate = cooperators / n_agents
    cooperation_rate_over_time.append(cooperation_rate)
    
    # Update agents' trust and Q-tables
    for i, agent in enumerate(agents):
        # Update trust based on group average contribution and trend
        group_avg_contribution = total_contributions / n_agents
        prev_trust = agent.trust
        agent.trust = lambda_trust * agent.trust + (1 - lambda_trust) * (group_avg_contribution / max(contribution_choices))
        
        # Trust bonus for increasing cooperation trends
        if t > 5:
            prev_avg = np.mean(contributions_over_time[-5:]) if len(contributions_over_time) >= 5 else c_mean
            if c_mean > prev_avg:
                agent.trust = min(agent.trust * 1.05, 1.0)  # Trust grows with increasing cooperation
        
        # Calculate total utility with behavioral factors
        u_beh = behavioral_utility(
            contributions[i], total_contributions, agent.trust, 
            payoffs[i], p_mean, c_mean, agents, i, agent
        )
        
        # Update Q-table using Q-learning
        action_idx = np.where(contribution_choices == contributions[i])[0][0]
        max_future_q = np.max(agent.q_table)
        agent.q_table[action_idx] = (1 - alpha_q) * agent.q_table[action_idx] + alpha_q * (u_beh + gamma_q * max_future_q)
    
    # Apply RLHF periodically
    if t % rlhf_frequency == 0:
        for agent in agents:
            for idx, c in enumerate(contribution_choices):
                # Progressive boost based on contribution level
                boost = rlhf_boost * (c / max(contribution_choices))**1.5
                agent.q_table[idx] += boost
    
    # Record metrics
    contributions_over_time.append(c_mean)
    trust_over_time.append(np.mean([agent.trust for agent in agents]))
    payoff_variance_over_time.append(np.var(payoffs))

print("Simulation complete.")

# Save Results to CSV
data = pd.DataFrame({
    'Round': range(n_rounds),
    'Average_Contribution': contributions_over_time,
    'Average_Trust': trust_over_time,
    'Payoff_Variance': payoff_variance_over_time,
    'Cooperation_Rate': cooperation_rate_over_time
})
data.to_csv('simulation_results_fair.csv', index=False)
print("Results saved to CSV.")

# Plot Results
sns.set_style("whitegrid")
sns.set_context("talk")
plt.figure(figsize=(12, 16))

# Average Contribution Plot
plt.subplot(4, 1, 1)
plt.plot(contributions_over_time, color='#1f77b4', linewidth=2)
plt.title('Average Contribution Over Time')
plt.xlabel('Round')
plt.ylabel('Average Contribution')
plt.ylim(0, 10)
plt.grid(True, alpha=0.3)

# Average Trust Plot
plt.subplot(4, 1, 2)
plt.plot(trust_over_time, color='#2ca02c', linewidth=2)
plt.title('Average Trust Over Time')
plt.xlabel('Round')
plt.ylabel('Average Trust')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Payoff Variance Plot
plt.subplot(4, 1, 3)
plt.plot(payoff_variance_over_time, color='#d62728', linewidth=2)
plt.title('Payoff Variance Over Time')
plt.xlabel('Round')
plt.ylabel('Payoff Variance')
plt.grid(True, alpha=0.3)

# Cooperation Rate Plot
plt.subplot(4, 1, 4)
plt.plot(cooperation_rate_over_time, color='#9467bd', linewidth=2)
plt.title('Cooperation Rate Over Time')
plt.xlabel('Round')
plt.ylabel('Proportion of Cooperators')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_plots_fair.png', dpi=300)
print("Plots saved.")
plt.show()