import numpy as np
import torch
from maraboupy import Marabou

def marbou_walks_down_to_holes_finder(self):
    print("here")
    states_to_fix = []
    options = Marabou.createOptions(verbosity=0)
    stateIndex = self.env.reset()
    new_state_arr = np.zeros(self.state_size)
    new_state_arr[stateIndex] = 1
    down_critical_states = self.env.return_all_down_critical_states()
    for stateIndex in down_critical_states:
        network = Marabou.read_onnx("q_network_init.onnx")
        arr = np.zeros(self.state_size)
        arr[stateIndex] = 1
        # adding down restrictions on output
        # x0 -x3 <0, x1 -x3 <0, x2 -x3 <0
        # if sat it means we walk into a hole
        for i in self.env.hole_index_list:
            network.setLowerBound(i,0)
            network.setUpperBound(i, 0)
        for i in range(self.size*self.size):
            if i not in self.env.hole_index_list:
                if i != stateIndex:
                    network.setLowerBound(i, 0)
                    network.setUpperBound(i, 0)
        network.setLowerBound(stateIndex, 1)
        network.setUpperBound(stateIndex, 1)
        network.addInequality(network.outputVars[0][0], [1, 0, 0, -1], 0)
        network.addInequality(network.outputVars[0][0], [0, 1, 0, -1], 0)
        network.addInequality(network.outputVars[0][0], [0, 0, 1, -1], 0)
        vals = network.solve(options=options)
        if vals is None:
            print("Something bad happened")
        else:
            print("Walked down to a hole? ", vals)
            if vals[0] == 'sat':
                states_to_fix.append(arr)
    print("done")
    return states_to_fix

def marbou_walks_up_to_holes_finder(self):
    print("here")
    states_to_fix = []
    options = Marabou.createOptions(verbosity=0)
    stateIndex = self.env.reset()
    new_state_arr = np.zeros(self.state_size)
    new_state_arr[stateIndex] = 1
    up_critical_states = self.env.return_all_up_critical_states()
    for stateIndex in up_critical_states:
        network = Marabou.read_onnx("q_network_init.onnx")
        arr = np.zeros(self.state_size)
        arr[stateIndex] = 1
        # adding up restrictions on output
        # if sat it means we walk into a hole
        for i in self.env.hole_index_list:
            network.setLowerBound(i,0)
            network.setUpperBound(i, 0)
        for i in range(self.size*self.size):
            if i not in self.env.hole_index_list:
                if i != stateIndex:
                    network.setLowerBound(i, 0)
                    network.setUpperBound(i, 0)
        network.setLowerBound(stateIndex, 1)
        network.setUpperBound(stateIndex, 1)
        network.addInequality(network.outputVars[0][0], [1, 0, -1, 0], 0)
        network.addInequality(network.outputVars[0][0], [0, 1, -1, 0], 0)
        network.addInequality(network.outputVars[0][0], [0, 0, -1, 1], 0)
        vals = network.solve(options=options)
        if vals is None:
            print("Something bad happened")
        else:
            print("Walked up to a hole? ", vals)
            if vals[0] == 'sat':
                states_to_fix.append(arr)
    print("done")
    return states_to_fix

def marbou_walks_left_to_holes_finder(self):
    print("here")
    states_to_fix = []
    options = Marabou.createOptions(verbosity=0)
    stateIndex = self.env.reset()
    new_state_arr = np.zeros(self.state_size)
    new_state_arr[stateIndex] = 1
    left_critical_states = self.env.return_all_left_critical_states()
    for stateIndex in left_critical_states:
        network = Marabou.read_onnx("q_network_init.onnx")
        arr = np.zeros(self.state_size)
        arr[stateIndex] = 1
        # adding left restrictions on output
        # x3 -x0 <0, x1 -x0 <0, x2 -x0 <0
        # if sat it means we walk into a hole
        for i in self.env.hole_index_list:
            network.setLowerBound(i,0)
            network.setUpperBound(i, 0)
        for i in range(self.size*self.size):
            if i not in self.env.hole_index_list:
                if i != stateIndex:
                    network.setLowerBound(i, 0)
                    network.setUpperBound(i, 0)
        network.setLowerBound(stateIndex, 1)
        network.setUpperBound(stateIndex, 1)
        network.addInequality(network.outputVars[0][0], [-1, 1, 0, 0], 0)
        network.addInequality(network.outputVars[0][0], [-1, 0, 1, 0], 0)
        network.addInequality(network.outputVars[0][0], [-1, 0, 0, 1], 0)
        vals = network.solve(options=options)
        if vals is None:
            print("Something bad happened")
        else:
            print("Walked left to a hole? ", vals)
            if vals[0] == 'sat':
                states_to_fix.append(arr)
    print("done")
    return states_to_fix
def marbou_walks_right_to_holes_finder(self):
    print("here")
    states_to_fix = []
    options = Marabou.createOptions(verbosity=0)
    stateIndex = self.env.reset()
    new_state_arr = np.zeros(self.state_size)
    new_state_arr[stateIndex] = 1
    right_critical_states = self.env.return_all_right_critical_states()
    for stateIndex in right_critical_states:
        network = Marabou.read_onnx("q_network_init.onnx")
        arr = np.zeros(self.state_size)
        arr[stateIndex] = 1
        # adding left restrictions on output
        # x3 -x0 <0, x1 -x0 <0, x2 -x0 <0
        # if sat it means we walk into a hole
        for i in self.env.hole_index_list:
            network.setLowerBound(i,0)
            network.setUpperBound(i, 0)
        for i in range(self.size*self.size):
            if i not in self.env.hole_index_list:
                if i != stateIndex:
                    network.setLowerBound(i, 0)
                    network.setUpperBound(i, 0)
        network.setLowerBound(stateIndex, 1)
        network.setUpperBound(stateIndex, 1)
        network.addInequality(network.outputVars[0][0], [1, -1, 0, 0], 0)
        network.addInequality(network.outputVars[0][0], [0, -1, 1, 0], 0)
        network.addInequality(network.outputVars[0][0], [0, -1, 0, 1], 0)
        vals = network.solve(options=options)
        if vals is None:
            print("Something bad happened")
        else:
            print("Walked right to a hole? ", vals)
            if vals[0] == 'sat':
                states_to_fix.append(arr)
    print("done")
    return states_to_fix