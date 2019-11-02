import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def draw_Q(Q, shape):

    best_q_values = np.zeros(shape)

    for (y,x), a in Q.keys():
        if best_q_values[y,x] == 0 or best_q_values[y,x] < Q[(y,x), a] and not Q[(y,x), a] == 0:
            best_q_values[y,x] = Q[(y,x), a]
    
    fig, ax = plt.subplots(figsize=(20,15))
    plt.title('State-Action Function', fontsize=30)

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    im = ax.imshow(best_q_values)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Value', rotation=-90, va='bottom', fontsize=20)

    return best_q_values


def plot_history(env, h):

    grid = np.ones(env.state_space)*255
    xs = []; ys = []; us = []; vs = []
    for t in range(1,len(h)-1):
        state = h[t][0]
        next_state = h[t+1][0]
        
        x = state[1]
        y = state[0]
        u = next_state[1] - state[1]
        v = -(next_state[0] - state[0])

        xs.append(x)
        ys.append(y)
        us.append(u)
        vs.append(v)

        grid[y,x] = 200

    for y,x in env.reset_state:
        grid[y,x] = 120

    fig, ax = plt.subplots(1, figsize=(20,15))
    plt.imshow(grid, cmap='gray', vmin=0, vmax=255)

    ax.grid(linestyle='-', linewidth=1)

    ax.set_xticks(np.arange(-.5, 12, 1))
    ax.set_yticks(np.arange(-.5, 4, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.set_xlabel('Wind (number of steps pushed North)', fontsize=30)

    ax.text(env.initial_state[1], env.initial_state[0], 'Start', ha="center", va="center", fontsize=30)
    ax.text(env.terminate_state[1], env.terminate_state[0], 'End', ha="center", va="center", fontsize=30)

    #ax.xaxis.set_major_formatter(ticker.NullFormatter())
    #ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, 10, 1)))
    #ax.xaxis.set_minor_formatter(ticker.FixedFormatter(np.abs(env.north_wind)))
    #ax.tick_params(which='minor', labelsize=20)

    #plt.arrow(5.5, 6, 0, -4, width=2, head_width=5, head_length=2, alpha=0.4)
    plt.quiver(xs, ys, us, vs)

    plt.title('Policy', fontsize=35)


def plot_train_history(h):

    data = [x for (x,y) in h]

    fig, ax = plt.subplots(1, figsize=(20,15))
    ax.set_xlabel('Timesteps', fontsize=30)
    ax.set_ylabel('Episodes', fontsize=30)
    ax.tick_params(labelsize=25)
    plt.title('Training Progress', fontsize=35)

    plt.plot(data, np.arange(len(data)))
