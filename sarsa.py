import numpy as np
from windy_cliffside import WindyCliffside
from collections import defaultdict
import pickle
from plotter import *

ACT_TO_IND = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3}
IND_TO_ACT = {0:(1,0), 1:(-1,0), 2:(0,1), 3:(0,-1)}


def get_q_state(Q, state, env):
    return lambda a : [Q[state,action] for action in np.arange(env.action_space)][a]

'''Function only handles reading from Q, not writing. Removes the need to call ACT_TO_IND[x] on every Q read'''
def create_q_reader(Q):
    return lambda s, a : Q[s, ACT_TO_IND[a]]


def sarsa(env, policy, epsilon=0.1, alpha=0.5, gamma=1, iterations=1000):

    Q = defaultdict(float)
    Q_reader = create_q_reader(Q)
    no_steps = 0
    history = []

    for episode in range(iterations):
        if episode % 10 == 0:
            print("Playing episode {} out of {}".format(episode, iterations))

        state = env.reset()
        action = policy(epsilon, env.get_actions(), get_q_state(Q, state, env))

        game_over = False
        while not game_over:
            next_state, reward, game_over, info = env.step(action)
            next_action = policy(epsilon, env.get_actions(), get_q_state(Q, next_state, env))

            Q[state, ACT_TO_IND[action]] += alpha*(reward + gamma*Q_reader(next_state, next_action) - Q_reader(state, action))

            state = next_state; action = next_action
            no_steps += 1
        
        history.append((no_steps, episode))
            
    return Q, np.array(history)


def behavior_policy(epsilon, actions, q_state_values):
    '''
    Select the a greedy action with prob 1-epsilon. Select a non-greedy action with prob epsilon
    '''
    action_ids = [ACT_TO_IND[a] for a in actions]            # index of actions
    state_values = np.array([q_state_values(i) for i in action_ids])   # state value of actions

    best_action     = np.random.choice([action_ids[i] for i in np.where(state_values == state_values.max())[0]]) # select all greedy actions
    random_action   = np.random.choice([action_ids[i] for i in np.where(action_ids != best_action)[0]]) # select non-greedy action

    # Explore or greedy action?
    action_id = np.random.choice([best_action, random_action], p=[1-epsilon, epsilon])

    return IND_TO_ACT[action_id]


def play_no_explore(env, policy, Q):
    history = []

    state = env.reset()
    action = policy(0, env.get_actions(), get_q_state(Q, state, env))
    history.append((state, action))

    game_over = False
    while not game_over:
        next_state, _, game_over, _ = env.step(action)
        next_action = policy(0, env.get_actions(), get_q_state(Q, next_state, env))
    
        state = next_state; action = next_action
        history.append((state, action))

    return history

    

if __name__ == "__main__":
    env = WindyCliffside()
    train_history = []

    if True: # Save time but using pre-trained Q values
        Q, train_history = sarsa(env, behavior_policy, iterations=10000)

        with open('Q_Sarsa.p', 'wb') as f:
            pickle.dump(Q, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Q_Sarsa.p', 'rb') as f:
        Q = pickle.load(f)

    draw_Q(Q, env.state_space)

    play_history = play_no_explore(env, behavior_policy, Q)
    plot_history(env, play_history)
    plot_train_history(train_history)

    plt.show()
    