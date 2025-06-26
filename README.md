# Synaptic RL: Spiking Neural Networks & Reinforcement Learning (Weeks 1–2)

## Overview

This repository documents my progress in understanding and implementing Spiking Neural Networks (SNNs) and Reinforcement Learning (RL), culminating in a biologically inspired integration of STDP (Spike-Timing Dependent Plasticity) with Q-learning. The work is organized by week and objective, with all code and notebooks included.

---

## Table of Contents

- [Background: What I Knew Before Starting](#background-what-i-knew-before-starting)
- [Progress & Improvements](#progress--improvements)
- [New Theory & Practical Knowledge Gained](#new-theory--practical-knowledge-gained)
- [Project Structure & Code Overview](#project-structure--code-overview)
- [References & Learning Resources](#references--learning-resources)

---

## Background: What I Knew Before Starting

- **Basic Python programming** and familiarity with Jupyter notebooks.
- Some exposure to **deep learning** (standard ANNs, CNNs) and basic PyTorch usage.
- High-level understanding of **reinforcement learning** (RL) concepts, but limited hands-on experience.
- No practical experience with **spiking neural networks (SNNs)** or biologically inspired learning rules like STDP.

---

## Progress & Improvements

- **Reinforcement Learning (RL):**

  - Gained hands-on experience with RL environments (OpenAI Gym, Atari, Box2D, custom environments).
  - Implemented and trained agents using popular RL algorithms (PPO, A2C, DQN) via Stable Baselines3.
  - Improved understanding of RL training loops, reward structures, and evaluation metrics.

- **Spiking Neural Networks (SNNs):**

  - Learned to use the `snntorch` library for building and training SNNs.
  - Implemented a basic SNN for binary classification (MNIST 0 vs 1).
  - Understood the differences between SNNs and traditional ANNs, including temporal dynamics and spike-based computation.

- **Biological Learning Rules & Integration:**
  - Explored and implemented a simplified STDP rule.
  - Successfully integrated STDP with Q-learning to create a hybrid agent for the Taxi-v3 environment.
  - Tuned hyperparameters and added diagnostics/logging for better training and evaluation.

---

## New Theory & Practical Knowledge Gained

- **SNN Fundamentals:**

  - How SNNs process information using spikes and membrane potentials.
  - The role of surrogate gradients in training SNNs with backpropagation.

- **STDP (Spike-Timing Dependent Plasticity):**

  - Biological motivation and mathematical formulation of STDP.
  - How STDP can be used to update synaptic weights based on spike timing.

- **Q-learning & RL Algorithms:**

  - The Q-learning update rule and its implementation from scratch.
  - Exploration strategies (epsilon-greedy, softmax) and their impact on learning.
  - The importance of reward shaping, episode length, and diagnostics in RL.

- **Integrating SNNs with RL:**

  - How to represent Q-values as synaptic weights in an SNN.
  - Combining STDP updates with temporal-difference (TD) learning for more biologically plausible RL agents.

- **Debugging & Optimization:**
  - Diagnosed and resolved environment and dependency issues (e.g., PyTorch Inductor, MSVC on Windows).
  - Used logging and progress tracking to debug and optimize training loops.

---

## Project Structure & Code Overview

### Week 1: Foundations & Setup

- **RL.ipynb**: Classic RL with CartPole using PPO and DQN. Demonstrates environment setup, training, evaluation, and model saving.
- **Project 1 - Breakout.ipynb**: RL agent for Atari Breakout using A2C, including environment setup and evaluation.
- **Project 2 - Self Driving.ipynb**: RL agent for CarRacing-v2 using PPO, with custom wrappers for action compatibility.
- **Project 3 - Custom Environment.ipynb**: Creation and training of a custom Gym environment (ShowerEnv) with PPO.

### Week 2: SNNs + RL Fundamentals

- **SNN_binary_classification.ipynb**: Implements a basic SNN for binary MNIST classification using `snntorch`. Includes data preparation, network definition, training, and evaluation.
- **Q_learning.py**: Standalone Q-learning agent for Taxi-v3. Implements tabular Q-learning with epsilon-greedy exploration and evaluation.
- **STDP_Q_learning_integration.ipynb**: Main notebook for integrating STDP with Q-learning. Features:
  - SNN-inspired Q-table as a PyTorch module.
  - Custom STDP update rule.
  - Combined Q-learning and STDP weight updates.
  - Extensive logging and diagnostics for training and evaluation.

---

## References & Learning Resources

### Spiking Neural Networks (SNNs)

- [GeeksforGeeks: Spiking Neural Networks](https://www.geeksforgeeks.org/spiking-neural-networks-in-deep-learning/)
- [CNVRG: Spiking Neural Networks](https://cnvrg.io/spiking-neural-networks/)
- [YouTube: SNNs Explained](https://www.youtube.com/watch?v=GTXTQ_sOxak)
- [snntorch Tutorials](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html), [Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)

### Reinforcement Learning (RL)

- [GeeksforGeeks: RL](https://www.geeksforgeeks.org/spiking-neural-networks-in-deep-learning/)
- [YouTube: RL Course](https://www.youtube.com/watch?v=Mut_u40Sqz4)
- [Medium: Q-learning for Beginners](https://medium.com/data-science/q-learning-for-beginners-2837b777741)
- [DataCamp: Q-learning Tutorial](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial)
- [YouTube: Q-learning Explained](https://www.youtube.com/watch?v=MSrfaI1gGjI)

### STDP + Q-learning Integration

- [Neuromatch: STDP + RL](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html)
- [YouTube: STDP + RL](https://youtu.be/xRkonYlbzjs)
- [ScienceDirect: STDP + RL Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608021003609)
- [NGC Learn: STDP Tutorial](https://ngc-learn.readthedocs.io/en/latest/tutorials/neurocog/mod_stdp.html)

---

## How to Run

1. Clone the repository and install dependencies (see individual notebooks for requirements).
2. Open the notebooks in Jupyter or VS Code.
3. Follow the instructions in each notebook to run experiments and view results.

---

## Final Notes

This repository demonstrates my progress from foundational RL and SNN concepts to the integration of biologically inspired learning rules with modern RL algorithms. All code is organized and commented for clarity. Please see individual notebooks for detailed explanations and results.

---
