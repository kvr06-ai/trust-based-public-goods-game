# Trust-Based Behavioral Public Goods Game Simulation

This project implements a multi-agent simulation of a repeated public goods game (PGG), designed to explore how integrating human behavioral factors into Artificial Intelligence (AI) agents influences cooperation, fairness, and stability in complex social dilemmas.

## Problem Addressed

Traditional Multi-Agent Systems (MAS) often rely on purely rational agent models. These models struggle to achieve sustainable cooperation and equitable outcomes in scenarios mimicking human interactions, as they fail to capture crucial behavioral drivers like trust dynamics, fairness perception, and social pressure. This limitation hinders the development of AI systems capable of effective collaboration in real-world settings.

## Proposed Solution: Behavioral Mechanism Design Framework

This simulation employs a novel **behavioral mechanism design framework** that integrates key human behavioral factors into agent decision-making:

*   **Trust Dynamics:** Trust evolves based on past group performance (contributions) and trends, influencing an agent's willingness to cooperate. It incorporates a memory parameter (`lambda_trust`) to weigh recent experiences.
*   **Fairness Perception (Inequity Aversion):** Agents are sensitive to how their payoffs and contributions compare to the group average. They experience disutility from both disadvantageous (`alpha`) and advantageous (`beta`) inequity, promoting fairer outcomes.
*   **Social Pressure:** Agents are incentivized (`gamma`) to conform to the group's average contribution level, encouraging collective action and discouraging free-riding.
*   **Punishment:** Cooperators can penalize agents contributing significantly below the average (less than 70%), directly discouraging free-riding (`punishment_strength`).
*   **Cooperation Rewards:** Consistent cooperation (contributing above a threshold) is rewarded (`reward_cooperation`), reinforcing pro-social behavior.
*   **Hybrid Learning (Q-Learning + RLHF):** Agents adapt their contribution strategies using Q-learning, guided by a utility function incorporating the above behavioral factors. Learning is periodically boosted via simulated Reinforcement Learning with Human Feedback (RLHF), promoting desirable cooperative behaviors (`rlhf_frequency`, `rlhf_boost`).

These factors modify the agents' utility calculations, moving beyond pure payoff maximization to incorporate social and psychological considerations.

## Methodology

*   **Simulation Environment:** A repeated public goods game with 100 heterogeneous agents over 500 rounds.
*   **Context:** Modeled after open-source software development dynamics, representing a scalable, real-world PGG scenario.
*   **Learning:** Agents use Q-learning (`alpha_q`, `gamma_q`) with epsilon-greedy exploration (`epsilon_initial`, `epsilon_final`, `decay_rate`) combined with periodic RLHF interventions.
*   **Metrics:** Tracks average contributions, trust levels, payoff variance (as a proxy for fairness/equity), and cooperation rates over time.

## Key Findings (from Simulation)

*   **Enhanced Cooperation & Stability:** The behavioral framework significantly improved cooperation rates (stabilizing near 80-100%) and contribution levels (around 6-8 units) compared to purely rational baselines.
*   **Robust Trust:** Average trust levels grew quickly and remained high (60-80%), driven by the integrated cooperation incentives.
*   **Persistent Fairness Challenge:** Despite mechanisms promoting fairness, payoff variance remained high and volatile, indicating a significant challenge in achieving equitable outcomes alongside high cooperation.
*   **Effective Incentives:** The combination of rewards (trust bonuses, cooperation streaks) and penalties (punishment, inequity aversion, social pressure) effectively maintained high cooperation, although the "stick" elements might contribute to payoff volatility.

## Implications

*   Integrating behavioral factors demonstrably enhances cooperation and stability in MAS compared to purely rational models.
*   This framework offers a path towards creating more adaptive, cooperative, and human-aligned AI ecosystems.
*   Achieving fairness concurrently with efficiency remains a critical challenge, requiring further refinement of behavioral mechanisms.
*   The findings have potential applications in designing MAS for logistics, R&D consortia, decentralized finance (DeFi), and collaborative platforms.

## Requirements

*   Python 3.8+
*   Dependencies listed in `requirements.txt`

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python simulation.py
```

The simulation will generate:

*   `simulation_results_fair.csv`: Contains the time series data of average contributions, trust levels, payoff variance, and cooperation rates.
*   `simulation_plots_fair.png`: Visualizations of the simulation metrics over time.

## Parameters

The simulation includes several configurable parameters in `simulation.py`:

### Game Structure

*   `n_agents`: Number of agents (default: 100)
*   `n_rounds`: Number of simulation rounds (default: 500)
*   `endowment`: Initial resources per agent (default: 10)
*   `r`: Public good multiplication factor (default: 200, yielding MPCR = r / n_agents = 2.0)

### Behavioral Parameters

*   `lambda_trust`: Trust memory parameter (weight of past trust vs. current performance) (default: 0.4)
*   `eta`: Trust weight in utility function (default: 2.0)
*   `alpha`: Fairness parameter for disadvantageous inequity aversion (default: 4.0)
*   `beta`: Fairness parameter for advantageous inequity aversion (default: 3.0)
*   `gamma`: Social pressure weight (conformity incentive) (default: 2.0)
*   `punishment_strength`: Reduction factor applied to punished agents' payoffs (default: 0.3)
*   `reward_cooperation`: Utility bonus for consistent cooperation (default: 0.5)

### Learning Parameters

*   `alpha_q`: Learning rate for Q-learning (default: 0.2)
*   `gamma_q`: Discount factor for Q-learning (default: 0.95)
*   `epsilon_initial`: Initial exploration rate (default: 0.2)
*   `epsilon_final`: Final exploration rate (default: 0.01)
*   `decay_rate`: Rate of decay for exploration (default: 0.01)
*   `rlhf_frequency`: How often RLHF intervention occurs (default: every 5 rounds)
*   `rlhf_boost`: RLHF utility boost factor for high contributors (default: 0.8)
*   `rlhf_initial_boost`: Initial RLHF boost to kickstart cooperation (default: 2.0)

## License

MIT License

## Author

Kaushik Rajan
- ORCID: [0009-0003-7574-2148](https://orcid.org/0009-0003-7574-2148)
- Blog: [Medium](https://medium.com/@kaushikvr06)


---

This `README.md` file provides a complete guide for setting up and running the simulation. It includes an introduction, setup instructions, simulation parameters, guidance on interpreting and sharing results, and licensing information, ensuring all content is presented in a clear and logical flow.





