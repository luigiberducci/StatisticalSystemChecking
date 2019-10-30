import numpy as np
import ipdb
import random as rand

# network parameters
NUMBER_NODES = 7
PSEND = 0.05
MASTER_PSEND = 0.5

# property: no collisions
# it happens if and only if the master sends a message, one of the 1layer node forward it and the other two stay silent
# the chance are: MASTER_PSEND*PSEND*(1-PSEND)*(1-PSEND)
# in the current config: 0.23*0.05*0.95*0.95 = 0.01037 ~= 1%

class Node:
    def __init__(self, psend, active_state):
        self.psend = psend
        self.active = active_state

    def step(self, my_buffer):
        forward =  0;
        # if ACTIVE and only 1 message, prob. forward
        if self.active == 1 and my_buffer == 1:
            #with prob psend send and go to sleep
            #with prob 1-psend go to sleep
            if rand.random() < self.psend:
                forward = 1
            self.active = 0
        elif self.active == 1 and my_buffer != 1:
            #collision: remain in listening, reset buffer
            self.active = 1
        return forward

class FloodingNetwork:
    def __init__(self, number_nodes, psend, master_psend):
        # Net State
        self.states = np.ones(number_nodes)     # node states
        self.buffers = np.zeros(number_nodes)   # buffer states
        # Auxiliar counters
        # self.active_nodes = np.sum(self.states)
        # self.flooding_messages = np.sum(self.buffers)
        # Network
        self.nodes = []
        for i in range(number_nodes):
            if i==0:    #Master node
                node = Node(master_psend, self.states[i])
                self.buffers[0] = 1 # to start the flooding, the master must have a message to forward
            else:
                node = Node(psend, self.states[i])
            self.nodes.append(node)

    def get_state(self):
        return np.vstack((self.states, self.buffers))

    def run(self):
        # node 0
        # ipdb.set_trace()
        send = self.nodes[0].step(self.buffers[0])
        self.states[0] = self.nodes[0].active
        for node in [1,2,3]:
            self.buffers[node] += send
        self.buffers[0] = 0 #reset buffer
        # print(self.get_state())
        # node 1
        send = self.nodes[1].step(self.buffers[1])
        self.states[1] = self.nodes[1].active
        for node in [4,5,6]:
            self.buffers[node] += send
        self.buffers[1] = 0 #reset buffer
        # print(self.get_state())
        # node 2
        send = self.nodes[2].step(self.buffers[2])
        self.states[2] = self.nodes[2].active
        for node in [4,5,6]:
            self.buffers[node] += send
        self.buffers[2] = 0 #reset buffer
        # print(self.get_state())
        # node 3
        send = self.nodes[3].step(self.buffers[3])
        self.states[3] = self.nodes[3].active
        for node in [4,5,6]:
            self.buffers[node] += send
        self.buffers[3] = 0 #reset buffer
        # print(self.get_state())
        # node 4 are terminal nodes, we don't care about their forwarding
        self.nodes[4].step(self.buffers[4])
        self.nodes[5].step(self.buffers[5])
        self.nodes[6].step(self.buffers[6])
        self.states[4] = self.nodes[4].active
        self.states[5] = self.nodes[5].active
        self.states[6] = self.nodes[6].active
        self.buffers[4] = 0 #reset buffer
        self.buffers[5] = 0 #reset buffer
        self.buffers[6] = 0 #reset buffer
        # print(self.get_state())

def main():
    net = FloodingNetwork(NUMBER_NODES, PSEND, MASTER_PSEND)
    for i in range(1000000):
        net.run()
        state = net.get_state()
        print(state)
        print("{} {}".format(np.sum(state[0]), np.sum(state[1])))

if __name__=="__main__":
    main()
