# PokerPal AI

A poker bot that uses DQN and NFSP reinforcement learning strategies. Utilizes the RLCard reinforcement learning toolkit to simulate Limit Texas Hold'em poker games and train models using DQN and NFSP. Also conducts a tournament between the two trained agents to compare strategy effectiveness.

## Set-Up

Install the RLCard package and the supported version of TensorFlow using directions from the [RLCard GitHub](https://github.com/datamllab/rlcard).

## Add Pre-Trained and Rule-Based Models

  * Clone PokerPal AI repository
  * Run training files `<nfsp.py>` and `<dqn.py>`
  * Add model folders with same names as the training files to the `<rlcard/pretrained>` folder
  * Copy contents of this repository's files `<__init__.py>` and `<pretrained_models.py>` into files of the same name
  * Run file `<play.py>` folder to load pre-trained models, play the agents against one another, and evaluate how they perform
  
## Editable Hyperparameters
`<nfsp.py>` and `<dqn.py>`:
  * Frequency of training agent
  * Frequency of evaluating model performance
  * Frequency of plotting reward in log
  * Number of games in each tournament
  * Number of training episodes

`<play.py>` and `<dqn.py>`
  * Number of training episodes
