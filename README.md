# Trust-Based Public Goods Game Simulation

This project implements a multi-agent simulation of a public goods game with trust dynamics, fairness considerations, and reinforcement learning. The simulation explores how different behavioral factors influence cooperation and social dynamics in group settings.

## Features

- Multi-agent simulation with Q-learning and epsilon decay
- Trust dynamics with memory and trend-based growth
- Dual fairness considerations (payoff equity and contribution equity)
- Social pressure effects to promote conformity
- Punishment mechanisms for free-riders
- Cooperation streak rewards system
- RLHF (Reinforcement Learning with Human Feedback) with progressive boosts
- Detailed metrics tracking and visualization including cooperation rates

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python simulation.py
```

The simulation will generate:
- `simulation_results_fair.csv`: Contains the time series data of average contributions, trust levels, payoff variance, and cooperation rates
- `simulation_plots_fair.png`: Visualizations of the simulation metrics over time

## Parameters

The simulation includes several configurable parameters:

### Game Structure
- `n_agents`: Number of agents in the simulation (default: 100)
- `n_rounds`: Number of simulation rounds (default: 500)
- `endowment`: Initial resources given to each agent (default: 10)
- `r`: Public good multiplication factor (default: 200, giving MPCR = 2.0)

### Behavioral Parameters
- `lambda_trust`: Trust memory parameter (default: 0.4)
- `eta`: Trust weight in utility function (default: 2.0)
- `alpha`: Fairness parameter for disadvantageous inequity (default: 4.0)
- `beta`: Fairness parameter for advantageous inequity (default: 3.0)
- `gamma`: Social pressure weight (default: 2.0)
- `punishment_strength`: Strength of punishment for free-riders (default: 0.3)
- `reward_cooperation`: Bonus for consistent cooperation (default: 0.5)

### Learning Parameters
- `alpha_q`: Learning rate for Q-learning (default: 0.2)
- `gamma_q`: Discount factor for Q-learning (default: 0.95)
- `epsilon_initial`: Initial exploration rate (default: 0.2)
- `epsilon_final`: Final exploration rate (default: 0.01)
- `decay_rate`: Rate of decay for exploration (default: 0.01)
- `rlhf_frequency`: Frequency of RLHF interventions (default: every 5 rounds)
- `rlhf_boost`: RLHF boost factor for high contributors (default: 0.8)
- `rlhf_initial_boost`: Initial boost to establish cooperation (default: 2.0)

## Fairness Mechanisms

The simulation implements multiple mechanisms to promote fairness:

1. **Punishment System**: Cooperators can punish free-riders who contribute less than 70% of the group average
2. **Contribution and Payoff Equity**: Agents are sensitive to inequity in both contributions and payoffs
3. **Cooperation Rewards**: Agents receive additional utility for consistently contributing above a threshold
4. **Trust Bonuses**: Trust grows faster when group cooperation is increasing

## License

MIT License

## Author

Kaushik Rajan
- ORCID: [0009-0003-7574-2148](https://orcid.org/0009-0003-7574-2148)
- Blog: [Medium](https://medium.com/@kaushikvr06)


---

This `README.md` file provides a complete guide for setting up and running the simulation. It includes an introduction, setup instructions, simulation parameters, guidance on interpreting and sharing results, and licensing information, ensuring all content is presented in a clear and logical flow.





