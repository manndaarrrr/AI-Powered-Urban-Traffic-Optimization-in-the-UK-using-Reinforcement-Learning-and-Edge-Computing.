# AI-Powered-Urban-Traffic-Optimization-in-the-UK-using-Reinforcement-Learning-and-Edge-Computing.
Copyright (c) 2026 Mandar Satpute. All Rights Reserved.

This project, including all source code, model weights, and documentation, is proprietary. It is published here strictly for educational and evaluation purposes (such as academic verification or portfolio review). You may NOT copy, modify, distribute, sub-license, or use this code, in whole or in part, for any commercial or non-commercial purpose without explicit written permission from the author.

Table of Contents
Overview
System Architecture
Key Features
Installation & Setup
Usage
Results & Performance
Repository Structure
Contact

 Overview
This repository hosts the implementation of my MSc dissertation titled "Optimizing Traffic Flow in Urban Areas Using Reinforcement Learning and SUMO".

The project addresses the critical issue of urban traffic congestion by replacing traditional static traffic signal timers with an adaptive Artificial Intelligence agent. By leveraging Deep Reinforcement Learning (DRL), specifically a Deep Q-Network (DQN), the system learns to dynamically adjust traffic light phases based on real-time vehicle queue lengths and waiting times.

The simulation is built upon a real-world road network extracted from Sheffield, UK, using OpenStreetMap (OSM) data and simulated within SUMO (Simulation of Urban MObility).

 System Architecture
The system follows a standard Reinforcement Learning cycle:

Environment (SUMO): Simulates the traffic physics, vehicle movements, and traffic lights.

State Observation: The agent extracts real-time data via the TraCI API, including the number of vehicles in incoming lanes, their speed, and the current traffic light phase.

Agent (DQN): A Deep Q-Network processes the state and selects the optimal action (Green/Red phase duration) to maximize the cumulative reward.

Action Execution: The chosen phase is applied to the traffic lights in the simulation.

Reward Calculation: The agent receives feedback based on the reduction in cumulative waiting time and total travel time.

Key Features
Custom DQN Agent: Implemented a multi-layer neural network using PyTorch with Experience Replay and Target Networks to stabilize training.

TraCI Integration: utilized the Traffic Control Interface (TraCI) Python API for granular, step-by-step control over the SUMO simulation.

Real-World Network: Modeled a specific intersection in Sheffield (Sheaf Street / Park Square) to ensure practical relevance.

Dynamic Reward Function: Designed a reward function that penalizes long queues and high cumulative waiting times to encourage efficient flow.

Comparative Analysis: Includes scripts to benchmark the AI agent against traditional static (fixed-time) traffic controllers.

 Installation & Setup
To review or run this simulation locally, ensure you have the following prerequisites:

1. Prerequisites
Python 3.8+

Eclipse SUMO: Download and install the latest version from the official SUMO website. Ensure the SUMO_HOME environment variable is set.

2. Install Dependencies
Clone the repository and install the required Python libraries:

Bash
pip install -r requirements.txt
Key dependencies include: torch, traci, sumolib, numpy, matplotlib, pandas.

 Usage
Training the Agent
To train the DQN agent from scratch using the Sheffield network:

Bash
python train_agent.py --episodes 100 --batch_size 32
This will save the trained model weights to the models/ directory.

Testing / Evaluation
To visualize the trained agent controlling traffic in the SUMO GUI:

Bash
python run_simulation.py --model models/dqn_final.pth --gui
📊 Results & Performance
The system was evaluated over 100 simulation episodes. The AI-driven approach demonstrated significant improvements over the baseline static controller:

Average Waiting Time: Reduced by approximately 25% compared to fixed-time signals.

Queue Length: Successfully prevented gridlock scenarios where static timers failed during peak traffic density.

Throughput: Increased the total number of vehicles clearing the intersection per hour.

(See docs/Dissertation.pdf for full statistical analysis and graphs.)

 
Contact
This project was developed by Mandar Satpute as part of an MSc dissertation at Sheffield Hallam University.

Connect on LinkedIn: www.linkedin.com/in/mandar-satpute-aa7991230 / mandarsatpute3333@gmail.com
