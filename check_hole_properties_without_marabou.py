import numpy as np
import torch
import random
def invalid_action(current_state_index, action,SIZE):
    if action == 0 and current_state_index % SIZE == 0:
        return True
    elif action == 1 and current_state_index % SIZE == (SIZE - 1):
        return True
    elif action == 2 and current_state_index < SIZE:
        return True
    elif (action == 3 and (
            SIZE * SIZE - SIZE) <= current_state_index <= SIZE * SIZE - 1):
        return True
    else:
        return False
def verify_and_fix_holes_without_marabou(self):
    reward_fix = -2
    # Right critical states
    right_critical_states = self.env.return_all_right_critical_states()
    for stateIndex in reversed(right_critical_states):
        if  (stateIndex not in self.env.hole_index_list):
            self.env.reset(stateIndex)
            arr = np.zeros(self.state_size)
            arr[stateIndex] = 1
            arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.q_network_init(arr).argmax()
                action = torch.tensor([[action]])

            next_state, reward, done, suc = self.env.stepWithRewardShapingTest(action.item())
            print("action.item()",action.item())
            if action.item() == 1:
                next_state_arr = np.zeros(self.state_size)
                next_state_arr[next_state] = 1
                # Turing the state from an int to a pytorch tensor
                ns = torch.tensor(next_state_arr, dtype=torch.float32).unsqueeze(0)

                # Turing the reward from an int to a pytorch tensor
                reward = torch.tensor(reward, dtype=torch.float32)

                # Update Q-value using the Q-learning update rule
                with torch.no_grad():
                    random_num = random.uniform(0, 1)
                    if random_num > 0.5:
                        target = reward_fix + self.gamma * torch.max(self.q_network_init(ns))
                    else:
                        target = reward_fix + self.gamma * torch.max(self.target_network(ns))

                current = (torch.max(self.q_network_init(arr)))
                # Calculating loss
                loss = self.criterion(current, target)
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer2.step()
                break

    # Left critical states
    left_critical_states = self.env.return_all_left_critical_states()
    for stateIndex in reversed(left_critical_states):
        if (stateIndex not in self.env.hole_index_list):
            self.env.reset(stateIndex)
            arr = np.zeros(self.state_size)
            arr[stateIndex] = 1
            arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.q_network_init(arr).argmax()
                action = torch.tensor([[action]])

            next_state, reward, done, suc = self.env.stepWithRewardShapingTest(action.item())
            print("action.item()", action.item())
            if action.item() == 0:
                next_state_arr = np.zeros(self.state_size)
                next_state_arr[next_state] = 1
                # Turing the state from an int to a pytorch tensor
                ns = torch.tensor(next_state_arr, dtype=torch.float32).unsqueeze(0)

                # Turing the reward from an int to a pytorch tensor
                reward = torch.tensor(reward, dtype=torch.float32)

                # Update Q-value using the Q-learning update rule
                with torch.no_grad():
                    random_num = random.uniform(0, 1)
                    if random_num > 0.5:
                        target = reward_fix + self.gamma * torch.max(self.q_network_init(ns))
                    else:
                        target = reward_fix + self.gamma * torch.max(self.target_network(ns))

                current = (torch.max(self.q_network_init(arr)))
                # Calculating loss
                loss = self.criterion(current, target)
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer2.step()
                break

    # Up critical states
    up_critical_states = self.env.return_all_up_critical_states()
    for stateIndex in reversed(up_critical_states):
        if (stateIndex not in self.env.hole_index_list):
            self.env.reset(stateIndex)
            arr = np.zeros(self.state_size)
            arr[stateIndex] = 1
            arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.q_network_init(arr).argmax()
                action = torch.tensor([[action]])

            next_state, reward, done, suc = self.env.stepWithRewardShapingTest(action.item())
            if action.item() == 2:
                next_state_arr = np.zeros(self.state_size)
                next_state_arr[next_state] = 1
                # Turing the state from an int to a pytorch tensor
                ns = torch.tensor(next_state_arr, dtype=torch.float32).unsqueeze(0)

                # Turing the reward from an int to a pytorch tensor
                reward = torch.tensor(reward, dtype=torch.float32)

                # Update Q-value using the Q-learning update rule
                with torch.no_grad():
                    random_num = random.uniform(0, 1)
                    if random_num > 0.5:
                        target = reward_fix + self.gamma * torch.max(self.q_network_init(ns))
                    else:
                        target = reward_fix + self.gamma * torch.max(self.target_network(ns))

                current = (torch.max(self.q_network_init(arr)))
                # Calculating loss
                loss = self.criterion(current, target)
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer2.step()
                break

    # Down critical states
    down_critical_states = self.env.return_all_down_critical_states()
    for stateIndex in reversed(down_critical_states):
        if (stateIndex not in self.env.hole_index_list):
            self.env.reset(stateIndex)
            arr = np.zeros(self.state_size)
            arr[stateIndex] = 1
            arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.q_network_init(arr).argmax()
                action = torch.tensor([[action]])

            next_state, reward, done, suc = self.env.stepWithRewardShapingTest(action.item())
            if action.item() == 3:
                next_state_arr = np.zeros(self.state_size)
                next_state_arr[next_state] = 1
                # Turing the state from an int to a pytorch tensor
                ns = torch.tensor(next_state_arr, dtype=torch.float32).unsqueeze(0)

                # Turing the reward from an int to a pytorch tensor
                reward = torch.tensor(reward, dtype=torch.float32)

                # Update Q-value using the Q-learning update rule
                with torch.no_grad():
                    random_num = random.uniform(0, 1)
                    if random_num > 0.5:
                        target = reward_fix + self.gamma * torch.max(self.q_network_init(ns))
                    else:
                        target = reward_fix + self.gamma * torch.max(self.target_network(ns))

                current = (torch.max(self.q_network_init(arr)))
                # Calculating loss
                loss = self.criterion(current, target)
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer2.step()
                break


    stateIndex = self.env.reset()
    path = []
    path.append(stateIndex)
    new_state_arr = np.zeros(self.state_size)
    new_state_arr[stateIndex] = 1
    # Turing the state from an int to a pytorch tensor
    state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
    done = False
    max_step_per_training_ep = self.size * self.size
    total_reward_for_ep = 0
    step_number = 0
    while (not done) and (step_number < max_step_per_training_ep):
        # while not done:
        step_number += 1

        with torch.no_grad():
            action = self.q_network_init(state).argmax()
            action = torch.tensor([[action]])

        next_state, reward, done, suc = self.env.stepWithRewardShapingTest(action.item())


        invalid = invalid_action(stateIndex, action.item(), self.env.SIZE)
        if (invalid):
            # stepped outside board
            next_state_arr = np.zeros(self.state_size)
            next_state_arr[next_state] = 1
            # Turing the state from an int to a pytorch tensor
            ns = torch.tensor(next_state_arr, dtype=torch.float32).unsqueeze(0)


            # Update Q-value using the Q-learning update rule
            with torch.no_grad():
                random_num = random.uniform(0, 1)
                if random_num > 0.5:
                    target = reward_fix + self.gamma * torch.max(self.q_network_init(ns))
                else:
                    target = reward_fix + self.gamma * torch.max(self.target_network(ns))

            current = (torch.max(self.q_network_init(state)))
            # Calculating loss
            loss = self.criterion(current, target)
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()
            break
        if (next_state in path):
                break
        path.append(next_state)
        new_state_arr = np.zeros(self.state_size)
        new_state_arr[next_state] = 1
        next_state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
        state = next_state