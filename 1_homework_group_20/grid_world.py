import numpy as np 

from collections import defaultdict 
from tqdm import tqdm

  

class GridWorld: 
    """a Gridworld consists of an mxn sized world and the agent-position within this grid. 
    The agent can typically choose between four actions: [LEFT, TOP, RIGHT, DOWN]. 
    Additionally, there is typically a goal - a specific tile, which the agent is supposed to reach. 
    When the agent reaches this tile, a reward is given and the environment is reset."""

    def __init__(self, width, height, start, goal, walls, traps): 
        """create a Grid World

        Args:
            width (int): The width of the grid world
            height (int): The hight of the grid world
            start (tuple): The start point for an agent 
            goal (tuple): The goal point for an agent 
            walls (list): The walls in the world which an agent can not pass 
            traps (list): The traps in the world on which an agent get penalized
        """

        self.width = width 
        self.height = height 
        self.start = start 
        self.goal = goal 
        self.walls = walls 
        self.traps = traps 

  

    def get_state_space(self): 
        """
        Returns:
            list: All states in the grid world 
        """

        return [(x, y) for x in range(self.width) for y in range(self.height)] 
  

    def get_action_space(self):
        """
        Returns:
            list: All actions which can be performed in the grid world 
        """

        return ["LEFT", "UP", "RIGHT", "DOWN"] 

  

    def transition(self, state, action): 

        """After an action is performed the new postion of the agent is returned. A move can not be perfomed on a wall, so if an action would lead
        to an agent to end up on a wall the agent will stay in his position.  

        Returns:
            tupel: new state coordinates
        """

        x, y = state 

        if action == "LEFT": 

            x = max(0, x - 1) 

        elif action == "UP": 

            y = max(0, y - 1) 

        elif action == "RIGHT": 

            x = min(self.width - 1, x + 1) 

        elif action == "DOWN": 

            y = min(self.height - 1, y + 1) 

  
        #keep the current stay if the action would end up on a wall
        if (x, y) in self.walls:

            return state 

        else: 

            return (x, y) 

  

    def reward(self, next_state): 
        """Computes the reward for a state. The Reward for an empty file = -0.5, for a trap or wall = -10 and the goal = 10

        Args:
            next_state (tuple): The next state on which the agent is about to step

        Returns:
            int: the reward of a state
        """

        if next_state in self.traps: 

            return -10 

        elif next_state in self.goal: 

            return 50 

        elif next_state in self.walls:

            return -10

        else: 
            #for an empty field
            return -0.5 

  
#policy
class SimpleAgent: 
    """A custom agent (i.e. a policy) to interact with your GridWorld. 
    It is stochastic with non-zero probability for every action at every state:
    walk in the direction of the goal-state with a probability of 0.8, otherwise act randomly

    """

    def __init__(self, grid_world): 
        """create an policy based on a GridWorld object

        Args:
            grid_world (GridWorld): An object of a GridWorld in whihc the agent acts
        """

        self.grid_world = grid_world 

  

    def get_action(self, state): 
        """Get the action the agent should perform as the next move based in the current state.
        With a 80% Probability the next action will be the best possible action to maximize the reward.
        With a 20% Probabilty the action will be another randomly choosed action 

        Args:
            state (tuple): The agent's current state

        Returns:
            string: the action the agent will perform next
        """

        actions = self.grid_world.get_action_space() 

        goal_direction = np.array(self.grid_world.goal) - np.array(state) 

        optimal_action = actions[np.argmax(np.abs(goal_direction))] 

  
        if np.random.rand() < 0.8: 

            return optimal_action 

        else: 
          
            return np.random.choice(actions) 


def run_episode(agent, grid_world): 
    """One run of the agent until reaching the goal of the grid world

    Args:
        agent (SimlpeAgent): The policy/agent which decides the next move
        grid_world (GridWorld): The grid world in which the agent acts

    Returns:
        list: A list of every states the agent visits
        list: A list of all the rewards the agent collects during its run
    """

    state = grid_world.start #the starting state of the agent
    states = [state] #save it as the first state visited 
    rewards = [] 

    while state != grid_world.goal: 

        action = agent.get_action(state) #get the next action to be performed

        next_state = grid_world.transition(state, action) #get the new state 

        reward = grid_world.reward(next_state) #get the reward for performing the action


        states.append(next_state) 
        rewards.append(reward) 


        state = next_state #set the new state as the current
    #print(states,rewards)

    return states, rewards 

  

def evaluate_policy(agent, grid_world, num_episodes=10, MC_algo = "Every"): 
    """For all states s, which have been reached at least once in these episodes, 
       calculate a MC-estimation of Vπ(s) of this state.

    Args:
        agent (SimpleAgent): The policy/agent which decides the next move
        grid_world (GridWorld): The grid world in which the agent acts
        num_episodes (int, optional): Number of runs a agent performes. Defaults to 10.
        MC_algo (string, optional) = Sets the algorithm used to calculate the Monte Carlo Policy Evaluation.
                  First Visit Monte Carlo: Average returns only for first time s is visited in an episode ("First").
                  Every visit Monte Carlo: Average returns for every time s is visited in an episode ("Every").
                  Default to "Every"

    Returns:
        dict: State values for every state in the last episode
    """

    state_values = defaultdict(float) 

    
    for _ in tqdm(range(num_episodes)): 

        states, rewards = run_episode(agent, grid_world) 

        if MC_algo == "First": 

            unique_states  = list(set(states))


            for state in unique_states: 

                #Average all returns after the first occurance of s in an episode.
                state_start_point = states.index(state) #first occurrence

                # If sliding the last, manually need to set the last value as slicing index, since len(reward[last index:]) → [] 
                if state == grid_world.goal :
                    state_start_point = -1

                state_values[state] += sum(rewards[state_start_point:])/ len(rewards[state_start_point:])


        #if MC_algo == "Every" or an invalid input: 
        else:
            #enumerate all states visited skipping the inital state
            for k, state in enumerate(states): 

                # If sliding the last, manually need to set the last value as slicing index, since len(reward[last index:]) → [] 
                if state == grid_world.goal :
                    k = -1

                #Average returns for every time s is visited in an episode.
                state_values[state] += sum(rewards[k:])  / len(rewards[k:])
     
    return state_values 
  
#main
grid_world = GridWorld(width=4, height=4, start=(0, 0), goal=(3, 3), walls=[(1, 1), (2, 1)], traps=[(3, 1)]) 
agent = SimpleAgent(grid_world) 

state_values = evaluate_policy(agent, grid_world, MC_algo= "Every") #evaluate a policy 

#visualize the policy for every state
for y in range(grid_world.height): 

    for x in range(grid_world.width): 

        state = (x, y) 

        print(f"{state}:", f"{state_values[state]:6.2f}", end=" ") 

    print()