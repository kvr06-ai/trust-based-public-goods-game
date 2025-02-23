import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_AGENTS = 100
NUM_ROUNDS = 500
MAX_CONTRIBUTION = 10
ALPHA = 2.0  # Increased fairness penalty (disadvantageous inequity), was 1.5
BETA = 1.0   # Increased fairness penalty (advantageous inequity), was 0.75
GAMMA = 1.0  # Increased social pressure weight, was 0.8
R = 75       # Reduced public good return (MPCR = 0.75), was 100
EPSILON = 0.05  # Reduced exploration rate, was 0.1
LAMBDA = 0.6    # Increased trust memory for slower updates, was 0.4
INITIAL_RLHF_BOOST = 0.1  # Reduced initial RLHF boost, was 0.3

# Initialize agents' Q-tables, contributions, trust, and tracking arrays
q_tables = np.zeros((NUM_AGENTS, MAX_CONTRIBUTION + 1))  # Q-values for each contribution level
contributions = np.random.randint(0, MAX_CONTRIBUTION + 1, NUM_AGENTS)
trust = np.zeros(NUM_AGENTS)  # Initial trust starts at 0
avg_contributions = []
avg_trust = []
payoff_variances = []

# Utility function with fairness, social pressure, and equity reward
def behavioral_utility(contribution, payoff, avg_payoff, avg_contribution):
    # Base payoff: keep what you didn't contribute + share of public good
    personal_payoff = MAX_CONTRIBUTION - contribution + payoff
    # Fairness utility (penalize inequity)
    if personal_payoff < avg_payoff:
        fairness_utility = -ALPHA * (avg_payoff - personal_payoff)
    else:
        fairness_utility = -BETA * (personal_payoff - avg_payoff)
    # Social pressure utility (align with group contribution)
    social_pressure_utility = -GAMMA * abs(contribution - avg_contribution)
    # Equity reward (reward closeness to average payoff)
    equity_reward = 0.5 * (1 - abs(personal_payoff - avg_payoff) / max(personal_payoff, avg_payoff, 1e-6))
    return personal_payoff + fairness_utility + social_pressure_utility + equity_reward

# Simulation loop
for round_num in range(NUM_ROUNDS):
    # Apply initial RLHF boost only at round 0
    if round_num == 0:
        q_tables += INITIAL_RLHF_BOOST * np.random.random(q_tables.shape)

    # Agents choose contributions
    for i in range(NUM_AGENTS):
        if np.random.random() < EPSILON:  # Exploration
            contributions[i] = np.random.randint(0, MAX_CONTRIBUTION + 1)
        else:  # Exploitation
            contributions[i] = np.argmax(q_tables[i])

    # Calculate public good and individual payoffs
    total_contribution = np.sum(contributions)
    public_good = (R / NUM_AGENTS) * total_contribution  # MPCR = R / NUM_AGENTS = 0.75
    payoffs = public_good / NUM_AGENTS  # Equal share of public good

    # Update trust and Q-tables
    avg_contribution = np.mean(contributions)
    avg_payoff = np.mean([MAX_CONTRIBUTION - c + payoffs for c in contributions])
    for i in range(NUM_AGENTS):
        # Calculate utility
        utility = behavioral_utility(contributions[i], payoffs, avg_payoff, avg_contribution)
        # Update Q-table (simplified Q-learning update)
        q_tables[i, contributions[i]] = (1 - 0.1) * q_tables[i, contributions[i]] + 0.1 * utility
        # Update trust based on contribution alignment
        trust[i] = LAMBDA * trust[i] + (1 - LAMBDA) * (1 - abs(contributions[i] - avg_contribution) / MAX_CONTRIBUTION)

    # Record metrics
    avg_contributions.append(avg_contribution)
    avg_trust.append(np.mean(trust))
    payoffs_array = np.array([MAX_CONTRIBUTION - c + payoffs for c in contributions])
    payoff_variances.append(np.var(payoffs_array))

# Plot results
plt.figure(figsize=(10, 12))

plt.subplot(3, 1, 1)
plt.plot(avg_contributions, color='blue')
plt.title("Average Contribution Over Time")
plt.xlabel("Round")
plt.ylabel("Average Contribution")
plt.ylim(0, MAX_CONTRIBUTION + 2)

plt.subplot(3, 1, 2)
plt.plot(avg_trust, color='green')
plt.title("Average Trust Over Time")
plt.xlabel("Round")
plt.ylabel("Average Trust")
plt.ylim(0, 1)

plt.subplot(3, 1, 3)
plt.plot(payoff_variances, color='magenta')
plt.title("Payoff Variance Over Time")
plt.xlabel("Round")
plt.ylabel("Payoff Variance")
plt.ylim(0, 10)

plt.tight_layout()
plt.show()