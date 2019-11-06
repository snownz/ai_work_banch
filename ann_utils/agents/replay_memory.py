import numpy as np
import random

class ReplayMemorySequence:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self,
            buffer_size,
            batch_size,            
            unusual_sample_factor=0.99
            ):
        """Initialize a ReplayBuffer object.
        Params
        ======        
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = [ ]
                
        self.buffer_size = buffer_size     
        self.batch_size = batch_size    
        self.unusual_sample_factor = unusual_sample_factor
    
    def add(self, state, action, reward, next_state, position):
        """Add a new experience to memory."""
        
        e = [ state.copy(),
              action.copy(),
              reward.copy(),
              next_state.copy(),
              position ]

        self.memory.append( e )
        
        if len( self.memory ) > self.buffer_size:            
            del self.memory[ 0 ]
  
    def balanced_sample(self):

        # PRIORITIZING THE UNUSUAL EXPERIENCES
        sorted_memory = sorted( self.memory, key = lambda exp: abs( np.mean( exp[ 2 ] ) ), reverse=True )
        p = np.array( [ self.unusual_sample_factor ** i for i in range(len(sorted_memory)) ] )
        p = p / sum(p)
        sample_idxs = random.choices( np.arange( len(sorted_memory) ), k = self.batch_size, weights=p )
        samples = [ sorted_memory[idx] for idx in sample_idxs ] 

        states         = []
        actions        = []
        rewards        = []
        next_states    = []
        pos            = []

        for exp in samples:                        
            states.append         ( exp[0] )           
            actions.append        ( exp[1] )
            rewards.append        ( exp[2] )
            next_states.append    ( exp[3] )
            pos.append            ( exp[4] )

        states         = np.array(states)
        actions        = np.array(actions)
        rewards        = np.array(rewards)
        next_states    = np.array(next_states)
        pos            = np.array(pos)

        return states, actions, rewards, next_states, pos

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class ReplayMemoryGPT:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self,
            buffer_size,
            batch_size,            
            unusual_sample_factor=0.99
            ):
        """Initialize a ReplayBuffer object.
        Params
        ======        
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = [ ]
                
        self.buffer_size = buffer_size     
        self.batch_size = batch_size    
        self.unusual_sample_factor = unusual_sample_factor
    
    def add(self, state, action, reward, next_state, past, position):
        """Add a new experience to memory."""
        
        e = [ state.copy(),
              action,
              reward,
              next_state.copy(),
              past,
              position ]

        self.memory.append( e )
        
        if len( self.memory ) > self.buffer_size:            
            del self.memory[ 0 ]
  
    def balanced_sample(self):

        # PRIORITIZING THE UNUSUAL EXPERIENCES
        sorted_memory = sorted( self.memory, key = lambda exp: abs( exp[ 2 ] ), reverse = True )
        p = np.array( [ self.unusual_sample_factor ** i for i in range(len(sorted_memory)) ] )
        p = p / sum(p)
        sample_idxs = random.choices( np.arange( len(sorted_memory) ), k = self.batch_size, weights=p )
        samples = [ sorted_memory[idx] for idx in sample_idxs ] 

        states         = []
        actions        = []
        rewards        = []
        next_states    = []
        past           = []
        pos            = []

        for exp in samples:                        
            states.append         ( exp[0] )           
            actions.append        ( exp[1] )
            rewards.append        ( exp[2] )
            next_states.append    ( exp[3] )
            past.append           ( exp[4] )
            pos.append            ( exp[5] )

        states         = np.array(states)
        actions        = np.array(actions)
        rewards        = np.array(rewards)
        next_states    = np.array(next_states)
        past           = np.array(past)
        pos            = np.array(pos)

        return states, actions, rewards, next_states, past, pos

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class ReplayMemoryLSTM:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, unusual_sample_factor=0.99):
        """Initialize a ReplayBuffer object.
        Params
        ======        
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = [ ]
                
        self.buffer_size = buffer_size     
        self.batch_size = batch_size    
        self.unusual_sample_factor = unusual_sample_factor
    
    def add(self, state, next_state, action, prior_action, reward, bonus, t_state, done):
        """Add a new experience to memory."""

        s_ = next_state
        if next_state is None:
            s_ = np.zeros_like( state )

        e = [ state, s_, action, prior_action, reward, bonus, t_state, done ]

        self.memory.append( e )
        
        if len( self.memory ) > self.buffer_size:            
            del self.memory[ 0 ]
  
    def balanced_sample(self):

        # PRIORITIZING THE UNUSUAL EXPERIENCES
        sorted_memory = sorted( self.memory, key = lambda exp: abs( exp[ 2 ] ), reverse=True )
        p = np.array( [ self.unusual_sample_factor ** i for i in range(len(sorted_memory)) ] )
        p = p / sum(p)
        sample_idxs = random.choices( np.arange( len(sorted_memory) ), k = self.batch_size, weights=p )
        samples = [ sorted_memory[idx] for idx in sample_idxs ] 

        states      = []
        next_states = []
        actions     = []
        p_actions   = []
        rewards     = []
        bonus       = []
        t_state     = []
        dones       = []

        for exp in samples:                        
            states.append     ( exp[0] )
            next_states.append( exp[1] )
            actions.append    ( exp[2] )
            p_actions.append  ( exp[3] )
            rewards.append    ( exp[4] )
            bonus.append      ( exp[5] )
            t_state.append    ( exp[6] )
            dones.append      ( exp[7] )

        states      = np.array(states)
        next_states = np.array(next_states)
        actions     = np.array(actions)
        p_actions   = np.array(p_actions)
        rewards     = np.array(rewards)
        bonus       = np.array(bonus)
        t_state     = np.array(t_state)
        dones       = np.array(dones)

        return states, next_states, actions, p_actions, rewards, bonus, t_state, dones, len(samples)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, unusual_sample_factor=0.99):
        """Initialize a ReplayBuffer object.
        Params
        ======        
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = [ ]
                
        self.buffer_size = buffer_size     
        self.batch_size = batch_size    
        self.unusual_sample_factor = unusual_sample_factor
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        len_next_state = 0 if next_state is None else len( next_state )  
        s_ = next_state
        if next_state is None:
            s_ = np.zeros_like( state )

        e = [ state, action, reward, s_, len_next_state, done ]

        self.memory.append( e )
        
        if len( self.memory ) > self.buffer_size:            
            del self.memory[ 0 ]

    def sample_abs(self):
        # PRIORITIZING THE UNUSUAL EXPERIENCES
        sorted_memory = sorted( self.memory, key=lambda exp: abs( exp[2] ), reverse=True )
        p = np.array( [ self.unusual_sample_factor ** i for i in range(len(sorted_memory)) ] )
        p = p / sum(p)
        sample_idxs = random.choices( np.arange( len(sorted_memory) ), k=self.batch_size, weights=p )
        samples = [ sorted_memory[idx] for idx in sample_idxs ] 

        states         = []
        actions        = []
        rewards        = []
        next_states    = []
        len_next_state = []
        dones          = []

        for exp in samples:                        
            states.append        ( exp[0] )           
            actions.append       ( exp[1] )
            rewards.append       ( exp[2] )
            next_states.append   ( exp[3] )
            len_next_state.append( exp[4] )
            dones.append         ( exp[5] )

        states         = np.array(states)
        actions        = np.array(actions)
        rewards        = np.array(rewards)
        next_states    = np.array(next_states)
        len_next_state = np.array(len_next_state)
        dones          = np.array(dones)

        return states, actions, rewards, next_states, len_next_state, dones

    def sample_inverse_dist(self):
        rewards_inverse_distribution = self.rewards_inverse_distribution()

        # PRIORITIZING THE UNUSUAL EXPERIENCES

        samples = []        
        rewards = [ k for k, v in rewards_inverse_distribution.items() ]
        probs = [ v for k, v in rewards_inverse_distribution.items() ]
        for _ in range( self.batch_size ):
            r_chosen = random.choices( rewards, weights=probs )[0]
            reward_exp = [ exp for exp in self.memory if exp[2] == r_chosen ]

            samples.append( random.choice( reward_exp ) )

        states         = []
        actions        = []
        rewards        = []
        next_states    = []
        len_next_state = []
        dones          = []


        for exp in samples:                        
            states.append        ( exp[0] )           
            actions.append       ( exp[1] )
            rewards.append       ( exp[2] )
            next_states.append   ( exp[3] )
            len_next_state.append( exp[4] )
            dones.append         ( exp[5] )

        states         = np.array(states)
        actions        = np.array(actions)
        rewards        = np.array(rewards)
        next_states    = np.array(next_states)
        len_next_state = np.array(len_next_state)
        dones          = np.array(dones)

        return states, actions, rewards, next_states, len_next_state, dones

    def rewards_distribution(self):
        reward_freq = {}

        for exp in self.memory:
            if exp[2] in reward_freq:
                reward_freq[ exp[2] ] += 1
            else:
                reward_freq[ exp[2] ] = 1
        
        reward_dist = {}
        for k, value in reward_freq.items():
            reward_dist[k] = value / len( self.memory )

        return reward_dist

    def rewards_inverse_distribution(self):
        reward_inverse_freq = {}

        for exp in self.memory:
            if exp[2] in reward_inverse_freq:
                reward_inverse_freq[ exp[2] ] -= 1
            else:
                reward_inverse_freq[ exp[2] ] = len( self.memory )
        
        total = 0
        for k, value in reward_inverse_freq.items():            
            total += value

        reward_inverse_dist = {}
        for k, value in reward_inverse_freq.items():            
            reward_inverse_dist[k] = value / total

        return reward_inverse_dist

    def balanced_sample(self):

        # PRIORITIZING THE UNUSUAL EXPERIENCES
        sorted_memory = sorted( self.memory, key = lambda exp: abs( exp[ 2 ] ), reverse=True )
        p = np.array( [ self.unusual_sample_factor ** i for i in range(len(sorted_memory)) ] )
        p = p / sum(p)
        sample_idxs = random.choices( np.arange( len(sorted_memory) ), k = self.batch_size, weights=p )
        samples = [ sorted_memory[idx] for idx in sample_idxs ] 

        states         = []
        actions        = []
        rewards        = []
        next_states    = []
        len_next_state = []
        dones          = []

        for exp in samples:                        
            states.append        ( exp[0] )           
            actions.append       ( exp[1] )
            rewards.append       ( exp[2] )
            next_states.append   ( exp[3] )
            len_next_state.append( exp[4] )
            dones.append         ( exp[5] )

        states         = np.array(states)
        actions        = np.array(actions)
        rewards        = np.array(rewards)
        next_states    = np.array(next_states)
        len_next_state = np.array(len_next_state)
        dones          = np.array(dones)

        return states, actions, rewards, next_states, len_next_state, dones

    def sample(self):

        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample( self.memory, k = self.batch_size )

        s              = [ e[ 0 ] for e in experiences ]
        actions        = [ e[ 1 ] for e in experiences ]
        rewards        = [ e[ 2 ] for e in experiences ]
        s_             = [ e[ 3 ] for e in experiences ]
        len_next_state = [ e[ 4 ] for e in experiences ]
        dones          = [ e[ 5 ] for e in experiences ]

        
        return s, actions, rewards, s_, len_next_state, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class MemoryRollout(object):
   
    def __init__(self):
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []        
        self.bonuses = []
        self.end_state = None

    def add(self, state, action, reward, value, terminal, features, bonus, end_state):

        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.bonuses += [bonus]
        self.end_state = end_state

    def extend(self, other):

        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        self.bonuses.extend(other.bonuses)
        self.end_state = other.end_state

    def size(self):
        return len( self.states )
