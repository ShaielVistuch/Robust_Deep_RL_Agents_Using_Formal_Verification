from action import Action
import numpy as np
from copy import deepcopy
import random

def addRewardShape(s, current_state_index, size):
    r_s, c_s = s //size, s%size
    r_ns,c_ns =current_state_index // size, current_state_index % size
    change = (r_ns -r_s) + (c_ns -c_s)
    return change


class Frozen_Lake_Environment:

    def __init__(self, size):
        self.current_state_index = 0
        # Initializing last state to be the Goal ('G')
        self.board = ['F']  # The board is used to store the states' type
        self.SIZE = size  # Length\width of the square-shaped board
        self.hole_index_list = []
        self.state_space = []
        self.possible_actions = np.array([Action.Left, Action.Right, Action.Up, Action.Down])

        # Initializing all states except the last one to be Frozen ('F')
        for i in range(1, self.SIZE * self.SIZE - 1):
            self.board.append('F')

        # Initializing last state to be the Goal ('G')
        self.board.append('G')

        for i in range(np.array(self.board).size):
            self.state_space.append(i)

    def print_environment_parameters(self):
        print("-----------self.SIZE:-------------")
        print(self.SIZE)
        print("-------------self.board:------------")
        print(self.board)
        print("---------self.hole_index_list:---------")
        print(self.hole_index_list)
        print("---------self.state_space:--------")
        print(self.state_space)
        print("--------self.possible_actions:--------")
        print(self.possible_actions)

    def add_hard_restrictions_to_env(self):
        for i in range(self.SIZE,self.SIZE * self.SIZE - 1):
            self.board[i] = 'H'
            self.hole_index_list.append(i)
        for i in range(self.SIZE * self.SIZE - 1):
            if ((i + 1) % self.SIZE) == 0:
                self.board[i] = 'F'
                try:
                    self.hole_index_list.remove(i)
                except:
                    pass
            if i < (self.SIZE - 1):  # First row doesn't have holes
                self.board[i] = 'F'

        self.board[0] = 'S'  # Upper-Left corner is reserved for Start
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_easy_restrictions_to_env(self):
        for i in range(self.SIZE - 1):
            rand = random.choice(self.state_space)
            self.board[rand] = 'H'
            self.hole_index_list.append(rand)
        rand = self.SIZE * self.SIZE - 1
        while rand == (self.SIZE * self.SIZE - 1) or (rand in self.hole_index_list):
            rand = random.choice(self.state_space)
            if rand in self.hole_index_list:
                rand = self.SIZE * self.SIZE - 1
            else:
                self.board[rand] = 'S'  # First corner is reserved for Start
                self.current_state_index = rand
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal




    def step(self, action_index):
        action = Action(action_index)

        # Checking if the action chosen is within the defined board
        if self.invalid_action(action):
            return self.current_state_index, -1  , False

        # Updating current state
        if action == Action.Left:
            self.current_state_index -= 1
        elif action == Action.Right:
            self.current_state_index += 1
        elif action == Action.Up:
            self.current_state_index -= self.SIZE
        else:
            self.current_state_index += self.SIZE

        state_type = self.board[self.current_state_index]

        # We reached the starting point. We need to continue -> returning False
        # if state_type == 'S':
        #     return self.current_state_index, -10 * self.SIZE * self.SIZE, False

        # We reached a frozen place. We need to continue -> returning False
        if state_type == 'F' or state_type == 'S':
            return self.current_state_index, -1/(2*self.SIZE*self.SIZE) , False

        # We reached the GOAL! -> Returning True
        elif state_type == 'G':
            return self.current_state_index, self.SIZE*self.SIZE, True

        # We reached a HOLE! -> Returning True
        else:
            return self.current_state_index, -1 , True

    def stepWithRewardShaping(self, action_index, s="stam"):
        action = Action(action_index)
        if s == "stam":
            s = self.current_state_index
        # Checking if the action chosen is within the defined board
        if self.invalid_action(action):
            #print("invalid", -2 * self.SIZE / 5)
            return self.current_state_index, -2 * self.SIZE / 5, False
            #return self.current_state_index, -self.SIZE / 5, True


        # Updating current state
        # Updating current state
        if action == Action.Left:
            self.current_state_index -= 1
        elif action == Action.Right:
            self.current_state_index += 1
        elif action == Action.Up:
            self.current_state_index -= self.SIZE
        else:
            self.current_state_index += self.SIZE

        rewardShape = addRewardShape(s, self.current_state_index,self.SIZE)
        #print("rewardShape, ", rewardShape)
        #print(action, s, self.current_state_index)
        state_type = self.board[self.current_state_index]

        # We reached the starting point. We need to continue -> returning False
        # if state_type == 'S':
        #     return self.current_state_index, -10 * self.SIZE * self.SIZE, False

        # We reached a frozen place. We need to continue -> returning False
        if state_type == 'F' or state_type == 'S':
            rewardShape = rewardShape -self.SIZE / 50
            #print('F ', rewardShape)
            return self.current_state_index, rewardShape, False

        # We reached the GOAL! -> Returning True
        elif state_type == 'G':
            rewardShape = rewardShape + self.SIZE
            #print('G ', rewardShape)
            return self.current_state_index,rewardShape , True

        # We reached a HOLE! -> Returning True
        else:
            rewardShape = rewardShape -self.SIZE / 5
            #print('H ', rewardShape)
            return self.current_state_index, rewardShape, True


    def invalid_action(self, action):
        if action == Action.Left and self.current_state_index % self.SIZE == 0:
            return True
        elif action == Action.Right and self.current_state_index % self.SIZE == (self.SIZE - 1):
            return True
        elif action == Action.Up and self.current_state_index < self.SIZE:
            return True
        elif (action == Action.Down and (
                self.SIZE * self.SIZE - self.SIZE) <= self.current_state_index <= self.SIZE * self.SIZE - 1):
            return True
        else:
            return False

    def print_on_board_current_state(self):
        #print(self.board)
        temp_board = deepcopy(self.board)
        for i in range(0, self.SIZE * self.SIZE):
            if self.current_state_index != i:
                if temp_board[i] == 'F':
                    print('+', end=" ")
                if temp_board[i] == 'H':
                    print('h', end=" ")
                if temp_board[i] == 'G':
                    print('G', end=" ")
                if temp_board[i] == 'S':
                    print('S', end=" ")
            else:
                print('A', end=" ")
            if i % self.SIZE == (self.SIZE - 1):
                print()

    def reset(self, index_start=None,random_start=True):

        #print(self.hole_index_list)
        for i in range(self.SIZE * self.SIZE - 1):
            self.board[i] = 'F'
        for k in self.hole_index_list:
            self.board[k] = 'H'
        if random_start == True:
            if index_start ==None:
                rand = self.SIZE * self.SIZE - 1
                while rand == (self.SIZE * self.SIZE - 1) :
                    rand = random.choice(self.state_space)
                    if rand in self.hole_index_list:
                        rand = self.SIZE * self.SIZE - 1
                    else:
                        self.board[rand] = 'S'  # reserved for Start
                        self.current_state_index = rand
            else:
                self.board[index_start] = 'S'  # reserved for Start
                self.current_state_index = index_start
        else:
            self.board[0] = 'S'  # reserved for Start
            self.current_state_index = 0
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal
        self.current_state_index = self.board.index('S')
        #print(self.board)
        return self.current_state_index


    def get_holes(self):
        return self.hole_index_list

    def get_possible_actions(self):
        return self.possible_actions

    def get_state_space(self):
        return self.state_space

    def get_random_action(self):
        return np.random.choice(self.possible_actions)





    def stepWithRewardShapingTest(self, action_index):
        action = Action(action_index)
        s = self.current_state_index
        # Checking if the action chosen is within the defined board
        if self.invalid_action(action):
            #print("invalid", self.current_state_index)
            return self.current_state_index, -2 * self.SIZE / 5, False , False


        # Updating current state
        if action == Action.Left:
            self.current_state_index -= 1
        elif action == Action.Right:
            self.current_state_index += 1
        elif action == Action.Up:
            self.current_state_index -= self.SIZE
        else:
            self.current_state_index += self.SIZE

        rewardShape = addRewardShape(s, self.current_state_index,self.SIZE)
        #print("rewardShape, ", rewardShape)
        #print(action, s, self.current_state_index)
        state_type = self.board[self.current_state_index]

        # We reached the starting point. We need to continue -> returning False
        # if state_type == 'S':
        #     return self.current_state_index, -10 * self.SIZE * self.SIZE, False

        # We reached a frozen place. We need to continue -> returning False
        if state_type == 'F' or state_type == 'S':
            rewardShape = rewardShape -self.SIZE / 50
            #print('F ', self.current_state_index)
            return self.current_state_index, rewardShape, False ,False

        # We reached the GOAL! -> Returning True
        elif state_type == 'G':
            rewardShape = rewardShape + self.SIZE
            #print('G ', self.current_state_index)
            return self.current_state_index,rewardShape , True ,True

        # We reached a HOLE! -> Returning True
        else:
            rewardShape = rewardShape -self.SIZE / 5
            #print('H ', self.current_state_index)
            return self.current_state_index, rewardShape, True ,False


    def return_all_down_critical_states(self):
        down_critical_states = []
        for i in self.hole_index_list:
            #print(i)
            if i < self.SIZE:
                pass
            else:
                down_critical_states.append(i-self.SIZE)
        l = []
        for i in down_critical_states:
            if (i in self.hole_index_list) or (i == (self.SIZE * self.SIZE - 1)):
                pass
            else:
                l.append(i)
        print("down_critical_states ", l)
        return l

    def return_all_up_critical_states(self):
        up_critical_states = []
        for i in self.hole_index_list:
            if i >= self.SIZE*self.SIZE-self.SIZE:
                pass
            else:
                up_critical_states.append(i + self.SIZE)
        l = []
        for i in up_critical_states:
            if (i in self.hole_index_list) or (i == (self.SIZE * self.SIZE - 1)):
                pass
            else:
                l.append(i)
        return l

    def return_all_right_critical_states(self):
        right_critical_states = []
        for i in self.hole_index_list:
            if i % self.SIZE ==0:
                pass
            else:
                right_critical_states.append(i -1)
        l = []
        for i in right_critical_states:
            if (i in self.hole_index_list) or (i == (self.SIZE * self.SIZE - 1)):
                pass
            else:
                l.append(i)
        return l

    def return_all_left_critical_states(self):
        left_critical_states = []
        for i in self.hole_index_list:
            if (i+1) % self.SIZE ==0:
                pass
            else:
                left_critical_states.append(i + 1)
        print("left_critical_states ", left_critical_states)
        l = []
        for i in left_critical_states:
            if (i in self.hole_index_list) or (i == (self.SIZE*self.SIZE-1)):
                pass
            else:
                l.append(i)

        print("left_critical_states ",l)
        return l

    def add_layout_3_1(self):
        self.hole_index_list.append(3)
        self.board[3] = 'H'
        self.hole_index_list.append(6)
        self.board[6] = 'H'
        self.hole_index_list.append(7)
        self.board[7] = 'H'
        self.board[0] = 'S'  # First corner is reserved for Start
        self.current_state_index = 1
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_layout_3_2(self):
        self.hole_index_list.append(1)
        self.board[1] = 'H'
        self.hole_index_list.append(4)
        self.board[4] = 'H'
        self.board[0] = 'S'  # First corner is reserved for Start
        self.current_state_index = 0
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_layout_3_3(self):
        self.hole_index_list.append(4)
        self.board[4] = 'H'
        self.board[0] = 'S'  # First corner is reserved for Start
        self.current_state_index = 0
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_layout_5_1(self):
        self.hole_index_list.append(5)
        self.board[5] = 'H'
        self.hole_index_list.append(6)
        self.board[6] = 'H'
        self.hole_index_list.append(7)
        self.board[7] = 'H'
        self.hole_index_list.append(8)
        self.board[8] = 'H'
        self.hole_index_list.append(10)
        self.board[10] = 'H'
        self.hole_index_list.append(11)
        self.board[11] = 'H'
        self.hole_index_list.append(12)
        self.board[12] = 'H'
        self.hole_index_list.append(13)
        self.board[13] = 'H'
        self.hole_index_list.append(15)
        self.board[15] = 'H'
        self.hole_index_list.append(16)
        self.board[16] = 'H'
        self.hole_index_list.append(17)
        self.board[17] = 'H'
        self.hole_index_list.append(18)
        self.board[18] = 'H'
        self.hole_index_list.append(20)
        self.board[20] = 'H'
        self.hole_index_list.append(21)
        self.board[21] = 'H'
        self.hole_index_list.append(22)
        self.board[22] = 'H'
        self.hole_index_list.append(23)
        self.board[23] = 'H'
        self.board[0] = 'S'  # First corner is reserved for Start
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_layout_5_2(self):
        self.hole_index_list.append(5)
        self.board[5] = 'H'
        self.hole_index_list.append(6)
        self.board[6] = 'H'
        self.hole_index_list.append(7)
        self.board[7] = 'H'
        self.board[0] = 'S'  # First corner is reserved for Start
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal


