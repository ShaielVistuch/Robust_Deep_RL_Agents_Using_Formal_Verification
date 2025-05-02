import numpy as np
import torch
from maraboupy import Marabou
import sys, os

def blockPrint():
    #print("blocked:---------------------------------------:")
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    #print("printed:---------------------------------------:")
    sys.stdout = sys.__stdout__

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




def marabou_loop_finder(self):
    #possible_initial_states = []
    path = []
    options = Marabou.createOptions(verbosity=0)
    # picking random starting space to search for loop
    stateIndex = self.env.reset()
    path.append(stateIndex)
    max_steps=0
    while max_steps < (self.env.SIZE*self.env.SIZE):
        max_steps = max_steps + 1
        new_state_arr = np.zeros(self.state_size)
        new_state_arr[stateIndex] = 1
        dummy_input = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)

        # Load the ONNX model into Marabou
        network = Marabou.read_onnx("q_network_init.onnx")
        print("dummy", dummy_input)
        vars = network.evaluate(dummy_input,options=options)

        print("Evaluate result ", "\n" , vars, "\n", vars[0], "\n", vars[0][0])
        max_index = np.argmax(vars)
        print("max_index (0-3)", max_index)
        stateIndex_new = stateIndex
        if max_index == 0:
            stateIndex_new -= 1
        elif max_index == 1:
            stateIndex_new += 1
        elif max_index == 2:
            stateIndex_new -= self.env.SIZE
        else:
            stateIndex_new += self.env.SIZE
        if stateIndex_new == (self.env.SIZE*self.env.SIZE-1):
            return None
        invalid = invalid_action(stateIndex,max_index,self.env.SIZE)
        if invalid:
            print("stepped outside board")
            return stateIndex
        if stateIndex_new in self.env.hole_index_list:
            return None
        else:
            if stateIndex_new in path:
                print("found loop!")
                path.append(stateIndex_new)
                print("path is ", path)
                #return stateIndex_new
                #return None
                return path
                #return None

            # Updating current state
            stateIndex = stateIndex_new
            path.append(stateIndex)

    print("|||||||||||||||||||||||||||||||||||||||||||||")


def marabou_optimal_path_finder(self, optimal_path):
    #possible_initial_states = []
    path = []
    options = Marabou.createOptions(verbosity=0)
    # picking random starting space to search for path
    stateIndex = self.env.reset()
    path.append(stateIndex)
    max_steps=0
    while max_steps < optimal_path:
        max_steps = max_steps + 1
        new_state_arr = np.zeros(self.state_size)
        new_state_arr[stateIndex] = 1
        dummy_input = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)

        # Load the ONNX model into Marabou
        network = Marabou.read_onnx("q_network_init.onnx")
        print("dummy", dummy_input)
        vars = network.evaluate(dummy_input,options=options)

        print("Evaluate result ", "\n" , vars, "\n", vars[0], "\n", vars[0][0])
        max_index = np.argmax(vars)
        print("max_index (0-3)", max_index)
        stateIndex_new = stateIndex
        if max_index == 0:
            stateIndex_new -= 1
        elif max_index == 1:
            stateIndex_new += 1
        elif max_index == 2:
            stateIndex_new -= self.env.SIZE
        else:
            stateIndex_new += self.env.SIZE
        if stateIndex_new == (self.env.SIZE*self.env.SIZE-1):
            return None
        invalid = invalid_action(stateIndex,max_index,self.env.SIZE)
        if invalid:
            print("stepped outside board")
            return None
        if stateIndex_new in self.env.hole_index_list:
            return None
        else:
            path.append(stateIndex_new)

            # Updating current state
            stateIndex = stateIndex_new
            path.append(stateIndex)
    if len(path)>optimal_path:
        return path
    return None
