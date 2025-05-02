# Robust_Deep_RL_Agents_Using_Formal_Verification
Official implementation of "Robust Deep Reinforcement Learning Using Formal Verification" paper (TASE'25).  <br/> <br/>

The Frozen Lake environment is a grid-based navigation challenge where the agent must move from a randomly chosen starting state to a predefined goal while avoiding holes. Movements are restricted to the four cardinal directions, and the success criterion is the percentage of starting states from which the goal is reached.

### Requirements
maraboupy 2.0.0 <br/>
matplotlib 3.9.2 <br/>
numpy 2.1.1 <br/>
onnx 1.16.2 <br/>
torch 2.4.1 

### Main Files
_verify_deep_q_learning_with_replay_with_reward_shaping.py_ - Simple DQN with a single network, replay buffer and reward shaping. <br/><br/>
_verify_double_deep_fixed_q_targets.py_ - Deep Q-learning with an added target network for
stabilization. <br/><br/>
_verify_double_deep_v2.py_ - Revised Double Deep Q-learning designed to mitigate overestimation bias.
_check_verification_group_1_in_model.py, check_verification_group_2_in_model.py_ - Responsible for verifying properties in the model. <br/><br/>
_Frozen_Lake_Environment.py_ - Implementation of the environment.<br/><br/>
_main.py_ - main file.
The method _train_ at the algorithms' files takes as input the maximum number of episodes as well as whether to use the developed verification-based algorithm. 
### Running the code
To compare the preformance of the developed verification-based backpropagation algorithm to the standard algorithms, follow these steps: <br/>
1) Pick the wanted algorithm by placing its name in the "runs" list in line  of main.py. <br/>
2) At main.py, add the wanted layout of your choice to the environment, and set the size of your board. The method _train_ recives as a parameter the maximum amount of episodes. Set the parameter max_time in the chosen algorithm's file to the number of seconds you wish the algorithm will run. <br/>
3) Run main.py

### Results
![image](https://github.com/user-attachments/assets/c70563c8-bb2a-48c5-a80f-1c246ddb806d) <br/>
Three trials were conducted for each combination of algorithm and layout, depicted above, to account for the randomness inherent in the algorithms (the networks' parameters, epsilon greeedy strategy, replay memory sampling, etc.). The averages of these trials are detailed below: <br/>
![image](https://github.com/user-attachments/assets/0a946584-23c2-4f01-9702-2352c09e8a61) <br/>
It is recommended to run several trials, to account for the randomness inherent in the algorithms. 
