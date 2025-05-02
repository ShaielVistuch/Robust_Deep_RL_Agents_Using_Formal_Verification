import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from ReplayMemory import ReplayMemory
from check_verification_group_2_in_model import marabou_loop_finder
from check_verification_group_1_in_model import marbou_walks_down_to_holes_finder, marbou_walks_up_to_holes_finder, \
    marbou_walks_left_to_holes_finder, marbou_walks_right_to_holes_finder
import sys, os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
class verify_double_deep_2015():
    def __init__(self, env, siz, env2, batch_size=16):
        # Exploration \ Exploitation parameters
        self.time_list = []
        self.epsilon = 1.0  # Exploration parameter
        self.max_epsilon = 1.0  # Max for exploration
        self.min_epsilon = 0.01  # Min for exploration
        self.decay_rate = 0.001  # Exponential decay factor
        self.Transition =namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        # Define the Q-network
        class QNetwork(nn.Module):
            def __init__(self, n_observations, n_actions):
                super(QNetwork, self).__init__()
                self.layer1 = nn.Linear(n_observations, 16)
                self.layer2 = nn.Linear(16, 8)
                self.layer3 = nn.Linear(8, n_actions)

            # Returns tensor
            def forward(self, x):
                x = F.relu(self.layer1(x))
                x = F.relu(self.layer2(x))
                return self.layer3(x)

        # Initialize the environment
        self.size = siz
        self.env = env
        self.env2 = env2
        self.state_size = self.size * self.size
        self.action_size = (self.env.get_possible_actions()).size

        # Initialize the Q-network, loss function, and optimizer
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.q_network_init = QNetwork(self.state_size, self.action_size)
        self.q_network_init.load_state_dict(self.q_network.state_dict())
        #torch.save(self.q_network.state_dict(), "Sample Networks/original_q_network_state_dict_v3.pt")
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.optimizer2 = optim.Adam(self.q_network_init.parameters(), lr=0.001)

        # Training parameters
        #self.num_episodes = 30000
        self.gamma = 0.8
        self.epsilon = 1

        # Reward list (for the Learning Curve plot)
        self.rewards_list = []
        self.rewards_list2 = []

        # Create a replay memory buffer
        self.BATCH_SIZE = batch_size
        self.memory = ReplayMemory(30000)
        self.memory2 = ReplayMemory(30000)
        self.start = []
        for i in range(self.size * self.size - 1):
            if i not in self.env.hole_index_list:
                self.start.append(i)


    def train(self, num_episodes,apply_verification_fix):
        self.q_network_init.load_state_dict(self.q_network.state_dict())
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.optimizer2 = optim.Adam(self.q_network_init.parameters(), lr=0.001)
        self.memory = ReplayMemory(30000)
        self.rewards_list = []
        self.gamma = 0.8
        self.epsilon = 1
        start_count = time.time()
        max_time = 20
        for episode in range(num_episodes):
            if ((time.time()-start_count) > max_time):

                sum = 0
                enablePrint()
                print("Episode number: ", episode)
                print(f'Running entire algorithm took: {time.time()-start_count} seconds.')
                if apply_verification_fix:
                    for t in self.time_list:
                        sum = sum + t
                    print(f'Running verification-fix method took an average of: {sum / len(self.time_list)} seconds per episode, and {sum} seconds total.')
                return self.rewards_list
            #enablePrint()
            #print("Episode number: ", episode)
            blockPrint()
            # Getting the state -> remember that the state is an integer
            stateIndex = self.env.reset()
            new_state_arr = np.zeros(self.state_size)
            new_state_arr[stateIndex] = 1
            self.dummy_input = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
            # Turing the state from an int to a pytorch tensor
            state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)

            done = False
            max_step_per_training_ep = self.size *self.size
            total_reward_for_ep = 0
            step_number = 0
            while (not done) and (step_number < max_step_per_training_ep):

                step_number += 1

                random_num = random.uniform(0, 1)
                # epsilon greedy policy
                if random_num > self.epsilon:
                    with torch.no_grad():
                        action = self.q_network_init(state).argmax()
                        action = torch.tensor([[action]])
                else:
                    action = torch.tensor([[self.env.get_random_action().value]], dtype=torch.long)


                next_state, reward, done = self.env.stepWithRewardShaping(action.item())

                new_state_arr = np.zeros(self.state_size)
                new_state_arr[next_state] = 1
                next_state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32)

                self.memory.push(state, action, next_state, reward)

                if len(self.memory) >= 4*self.BATCH_SIZE:


                    self.epsilon = (self.max_epsilon - self.min_epsilon) * np.exp(
                        -self.decay_rate * episode) + self.min_epsilon

                    Transition, transitions = self.memory.sample(self.BATCH_SIZE)
                    # Transpose the batch
                    batch = self.Transition(*zip(*transitions))

                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                            batch.next_state)), dtype=torch.bool)# device=device,
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                       if s is not None])
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)

                    state_action_values = self.q_network_init(state_batch).gather(1, action_batch)

                    with torch.no_grad():
                        tn = torch.reshape((self.q_network_init(non_final_next_states).argmax(1)),(16,1))
                        next_state_values = self.target_network(non_final_next_states).gather(1, tn)

                    expected_state_action_values = (next_state_values * self.gamma) + torch.reshape(reward_batch,(16,1))

                    # Calculating loss
                    loss = self.criterion(state_action_values, expected_state_action_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                    i = 0
                    while i < (2*self.BATCH_SIZE ):
                        self.memory.remove()
                        i = i +1

                state = next_state
                total_reward_for_ep += reward.item()

            if apply_verification_fix:
                start = time.time()
                self.verify_and_fix()
                self.time_list.append(time.time() - start)
            self.rewards_list.append(total_reward_for_ep)

            if episode%5 == 0:
                self.target_network.load_state_dict(self.q_network_init.state_dict())

    def test_model(self, network, i):
        #print(i)
        # Getting the state -> remember that the state is an integer
        stateIndex = self.env.reset(i)
        print(stateIndex)
        new_state_arr = np.zeros(self.state_size)
        new_state_arr[stateIndex] = 1
        # Turing the state from an int to a pytorch tensor
        state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
        suc = False
        done = False
        max_step_per_training_ep = self.size * self.size
        total_reward_for_ep = 0
        step_number = 0
        while (not done) and (step_number < max_step_per_training_ep):
            # while not done:
            step_number += 1

            with torch.no_grad():
                action = network(state).argmax()
                action = torch.tensor([[action]])

            next_state, reward, done, suc = self.env.stepWithRewardShapingTest(action.item())

            new_state_arr = np.zeros(self.state_size)
            new_state_arr[next_state] = 1
            next_state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32)
            state = next_state
            total_reward_for_ep += reward.item()
        print("Finished test")
        self.env.print_on_board_current_state()
        return suc

    def plot_learning_curve(self,verification):
        sum = 0
        for num in self.rewards_list:
            sum = sum + num
        print("Average reward: " + str(sum / len(self.rewards_list)))
        # Plotting the learning curve
        x = [x for x in range(len(self.rewards_list))]
        plt.figure()
        plt.xlabel("Episode #")
        plt.ylabel("Reward per episode")

        # Take 'window' episode averages and plot them too
        data = np.array(self.rewards_list)
        window = 10
        average_y = []
        for ind in range(len(data) - window + 1):
            average_y.append(np.mean(data[ind:ind + window]))
        for ind in range(window - 1):
            average_y.insert(0, np.nan)
        if verification:
            plt.title("Learning Curve with verification fix\nDEEP Q-ALGO with Replay Memory and Reward Shaping")
        else:
            plt.title("Learning Curve without verification fix\nDEEP Q-ALGO with Replay Memory and Reward Shaping")
        plt.plot(x, average_y, 'r.-', label='10-episode reward average')
        plt.legend()
        plt.show(block=True)

    def verify_and_fix(self):
        torch.onnx.export(self.q_network_init, self.dummy_input, "q_network_init.onnx", input_names=["input"],
                          output_names=["output"])
        for co in range(self.size):

            states_to_fix = []
            states_to_fix = marbou_walks_down_to_holes_finder(self)
            if states_to_fix == []:
                states_to_fix = marbou_walks_up_to_holes_finder(self)
            else:
                self.multi_verification_fix(states_to_fix)
                states_to_fix = marbou_walks_up_to_holes_finder(self)
            if states_to_fix == []:
                states_to_fix = marbou_walks_left_to_holes_finder(self)
            else:
                self.multi_verification_fix(states_to_fix)
                states_to_fix = marbou_walks_left_to_holes_finder(self)
            if states_to_fix == []:
                states_to_fix = marbou_walks_right_to_holes_finder(self)
            else:
                self.multi_verification_fix(states_to_fix)
                states_to_fix = marbou_walks_right_to_holes_finder(self)


            if states_to_fix == []:
                states_to_fix = marabou_loop_finder(self)
                try:
                    for state in states_to_fix:
                        print(state)
                        self.multi_verification_fix(state, -0.5)
                    # states_to_fix = states_to_fix[0]
                except:
                    print("single")
                if states_to_fix != None:
                    a = np.zeros(self.state_size)
                    a[states_to_fix] = 1
                    states_to_fix = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                    self.multi_verification_fix(states_to_fix)
            else:
                self.multi_verification_fix(states_to_fix)
                states_to_fix = marabou_loop_finder(self)
                try:
                    states_to_fix = states_to_fix[0]
                except:
                    print("single")
                if states_to_fix != None:
                    a = np.zeros(self.state_size)
                    a[states_to_fix] = 1
                    states_to_fix = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                    self.multi_verification_fix(states_to_fix)

    def multi_verification_fix(self,states_to_fix,reward = -1):



        if states_to_fix != []:

            try:
                state = states_to_fix[len(states_to_fix)-1]
            except:
                state = states_to_fix

            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            next_state_arr = np.zeros(self.state_size)
            with torch.no_grad():
                q_values = self.q_network_init(s)

                action = torch.argmax(q_values).item()
            next_state2, reward2, done2 = self.env2.stepWithRewardShaping(action, np.where(state == 1)[0][0])
            next_state_arr[next_state2] = 1

            # Turing the state from an int to a pytorch tensor
            ns = torch.tensor(next_state_arr, dtype=torch.float32).unsqueeze(0)

            # Turing the reward from an int to a pytorch tensor
            reward = torch.tensor(reward, dtype=torch.float32)

            # Update Q-value using the Q-learning update rule
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.q_network_init(ns))

            current = (torch.max(self.q_network_init(s)))
            # Calculating loss
            loss = self.criterion(current, target)
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()

        torch.onnx.export(self.q_network_init, self.dummy_input, "q_network_init.onnx", input_names=["input"],
                          output_names=["output"])