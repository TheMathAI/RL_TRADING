# Deep Reinforcement Learning for Stock Trading

This repository contains the implementation of various deep reinforcement learning models, including Deep Q-Network (DQN), Double DQN (DDQN), and Dueling Double DQN (Dueling DDQN), developed to optimize stock trading strategies.

## Getting Started

These instructions will guide you on how to set up and run the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, the project requires the following Python packages:

- NumPy
- PyTorch
- gymnasium
- gym-anytrading
- yfinance
- pandas
- pandas_datareader
- matplotlib

You can install them using pip:

pip install numpy torch gymnasium gym-anytrading yfinance pandas pandas_datareader matplotlib

## Code Structure:

QNetwork - A basic deep Q-network for estimating action values.
DuelingQNetwork - Implements the Dueling Network Architecture, which separately estimates the state value and advantages for each action.
ReplayMemory - Manages a memory buffer that stores transitions collected from the environment.
DQN/ DDQN Classes - Handle the training logic for DQN and DDQN models, including action selection, learning, updating the target network, and decaying epsilon.

The training/testing functions follow in the notebook.
