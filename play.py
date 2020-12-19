import rlcard
from rlcard import models
import os
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('limit-holdem', config={'seed': 0})
evaluate_num = 1000
tournament_run = 1

# The paths for saving the logs of how our model performs in a tournament
log_dir = './experiments/tournament/'
print(os.getcwd())
#dqn_run = 4
#limit_holdem_dqn_model = models.load('models/nolimit_holdem_dqn/run' + str(dqn_run)) 
model_1 = models.load('limit-holdem-nfsp-run3')
model_2 = models.load('limit-holdem-dqn-run2')
#nfsp_run = 9
#limit_holdem_nfsp_model = models.load('models/nolimit_holdem_nfsp/run' + str(nfsp_run)) 
#limit_holdem_nfsp_model = models.load('models/nolimit_holdem_nfsp/run' + str(nfsp_run))


env.set_agents([model_1.agents[0], model_2.agents[0]])
print("env agents have been set!")
# Init a Logger to plot the learning curve
logger = Logger(log_dir + "run" + str(tournament_run))
# List of average payoffs for each play over evaluate_num tournament runs
payoffs = tournament(env, evaluate_num)

print(f"model_1 payoff: {payoffs[0]}")
print(f"model_2 payoff: {payoffs[1]}")

# Logs the average payoff of limit_holdem_dqn_model_1
logger.log_performance(env.timestep, payoffs)
# Close files in the logger
logger.close_files()
