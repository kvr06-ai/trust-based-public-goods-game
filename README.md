# Trust-Based Public Goods Game Simulation

This project implements a multi-agent simulation of a public goods game with trust dynamics, fairness considerations, and reinforcement learning. The simulation explores how different behavioral factors influence cooperation and social dynamics in group settings.

## Features

- Multi-agent simulation with Q-learning
- Trust dynamics with memory
- Fairness considerations (advantageous and disadvantageous inequity)
- Social pressure effects
- RLHF (Reinforcement Learning with Human Feedback) integration
- Detailed metrics tracking and visualization

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
- `simulation_results.csv`: Contains the time series data of average contributions, trust levels, and payoff variance
- `simulation_plots.png`: Visualizations of the simulation metrics over time

## Parameters

The simulation includes several configurable parameters:

- `n_agents`: Number of agents in the simulation
- `n_rounds`: Number of simulation rounds
- `endowment`: Initial resources given to each agent
- `r`: Public good multiplication factor
- `lambda_trust`: Trust memory parameter
- `eta`: Trust weight
- `alpha`: Fairness parameter (disadvantageous inequity)
- `beta`: Fairness parameter (advantageous inequity)
- `gamma`: Social pressure weight

## License

MIT License

## Author

Kaushik Rajan
- ORCID: [0009-0003-7574-2148](https://orcid.org/0009-0003-7574-2148)
- Blog: [Medium](https://medium.com/@kaushikvr06)


---

This `README.md` file provides a complete guide for setting up and running the simulation. It includes an introduction, setup instructions, simulation parameters, guidance on interpreting and sharing results, and licensing information, ensuring all content is presented in a clear and logical flow.





