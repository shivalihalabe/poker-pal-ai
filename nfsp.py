'''
Training an NFSP Agent to play Limit Texas Holdem
'''

import tensorflow as tf
import os

import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# path to save logs and learning curves to
log_dir = './experiments/limit_holdem_nfsp_result/'

# Train the agent every X steps
train_every = 64

# Make environment
env = rlcard.make('limit-holdem', config={'seed': 0})
eval_env = rlcard.make('limit-holdem', config={'seed': 0})

# frequency with which we evaluate performance of our model
evaluate_every = 300 # how often we want to plot the reward on the graph
evaluate_num = 100 # number of games in the tournament
episode_num = 30000

# Experiment #
run = 3

# initial memory allocated
memory_init_size = 1000

# set global seed
set_global_seed(0)


with tf.Session() as sess:

    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(sess,
                          scope='nfsp' + str(i),
                          action_num=env.action_num,
                          state_shape=env.state_shape,
                          hidden_layers_sizes=[512, 512],
                          anticipatory_param=0.1,
                          min_buffer_size_to_learn=memory_init_size,
                          q_replay_memory_init_size=memory_init_size,
                          train_every=train_every,
                          q_train_every=train_every,
                          q_mlp_layers=[512, 512])
        agents.append(agent)
    random_agent = RandomAgent(action_num=eval_env.action_num)

    env.set_agents(agents)
    eval_env.set_agents([agents[0], random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir + "run" + str(run))

    # Write model parameters in the logger
    logger.log("memory_init_size: " + str(memory_init_size))
    logger.log("train_every: " + str(memory_init_size))
    logger.log("evaluate_every: " + str(evaluate_every))
    logger.log("episode_num: " + str(episode_num))

    for episode in range(episode_num):

        # First sample a policy for the episode
        for agent in agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for trajectory in trajectories[i]:
                agents[i].feed(trajectory)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])


    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('NFSP')

    # Save model
    save_dir = 'models/limit_holdem_nfsp/run' + str(run)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
