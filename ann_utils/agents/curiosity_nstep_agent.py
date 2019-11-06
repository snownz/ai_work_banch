import sys
sys.path.append('../../')

import numpy as np

from ann_utils.agents.replay_memory import ReplayMemory

class Agent:
    def __init__(
            self,
            n_a,
            n_s,
            sess,
            model,            
            lr=0.0001,
            gamma=0.95,
            soft_update=0.95,
            epsilon=1.,
            replace_target_iter=300,
            curiosity_step=100,
            memory_size=10000,
            batch_size=128,
            n_step_return=5
    ):
        self.n_a = n_a
        self.n_s = n_s
        self.lr = lr
        self.gamma = gamma
        self.soft_update = soft_update
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.curiosity_step = curiosity_step
        self.n_step_return = n_step_return

        # total learning step
        self.learn_step_counter = 0

        self.memory = { i : None for i in range( self.memory_size ) }
        self.model = model
        self.index = 0
        
        self.memory = []
        self.R = 0.

        self.sess = sess

        self.optimizer_memory = ReplayMemory( memory_size, batch_size ) 

    def build_agent_brain(self):
        
        self.st, self.ac, self.er, self.s_, self.dn, \
        self.pred_train, self.dqn_train, self.q, \
        self.soft_update_op, self.dqn_loss = \
            self.model.build_training_graph( self.n_s, ( self.gamma ** self.n_step_return ), self.lr, self.soft_update )

    def get_sample(self, memory, n):  

        s, a, r, _  = self.memory[0]
        _, _, _, s_ = self.memory[n-1]
        
        return s, a, self.R, s_

    def step(self, state, action, reward, next_state, done):
        
        a_cats = np.zeros( self.n_a )	
        a_cats[ action ] = 1 
       
        self.memory.append( ( state, a_cats, reward, next_state ) )

        self.R = ( self.R + reward * ( self.gamma ** self.n_step_return ) ) / self.gamma

        if done or len( self.memory ) == self.n_step_return:

            while len( self.memory ) > 1:

                n = len( self.memory )
                s, a, r, s_ = self.get_sample( self.memory, n )

                self.optimizer_memory.add( s, a, r, s_, s_ is None )

                self.R = ( self.R - self.memory[0][2] ) / self.gamma
                self.memory.pop(0)		

            self.R = 0
 
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess( self.q, { self.st: s } )
            action = np.argmax( actions_value )
        else:
            action = np.random.randint( 0, self.n_a )
        return action

    def learn(self):

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.soft_update_op )

        if len( self.optimizer_memory.memory ) > self.batch_size:
            
            states, actions, rewards, next_states, _, dones = self.optimizer_memory.sample()
            actions = np.argmax( actions, axis = 1 )
            
            _, loss = self.sess( [ self.dqn_train, self.dqn_loss ], 
                                 { 
                                     self.st: states, 
                                     self.ac: actions, 
                                     self.er: rewards, 
                                     self.s_: next_states,
                                     self.dn: dones
                                  } )
        
            # delay training in order to stay curious
            if self.learn_step_counter % self.curiosity_step == 0:   
                self.sess( self.pred_train, { self.s_: next_states } )
        
            self.learn_step_counter += 1

            return loss        
            
        return 1