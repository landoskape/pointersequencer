# mainExperiment at checkpoint 1
import sys
import os

# add path that contains the dominoes package
mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

# standard imports
from copy import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda as torchCuda

# dominoes package
from dominoes import leagueManager as lm
from dominoes import agents as da

# input arguments
parser = argparse.ArgumentParser(description="Run dominoes experiment.")
parser.add_argument("-np", "--num-players", type=int, default=4, help="the number of players for each game")
parser.add_argument("-hd", "--highest-dominoe", type=int, default=9, help="highest dominoe value in the set")
parser.add_argument("-ng", "--num-games", type=int, default=10000, help="how many games to play to estimate ELO")
# note: ELO is probability based, so increasing the number of rounds will usually exaggerate differences in ELO
parser.add_argument("-nr", "--num-rounds", type=int, default=None, help="how many rounds to play for each game")
# adding copies of agents allows ELO to include estimates based on agents playing with themselves in the league -
# this is important because the game dynamics change depending on what agents are present
parser.add_argument("-ne", "--num-each", type=int, default=4, help="how many copies of each agent type to include in the league")
parser.add_argument("-fe", "--fraction-estimate", type=float, default=0.05, help="final fraction of elo estimates to use")
parser.add_argument("--nosave", default=False, action="store_true")

args = parser.parse_args()
assert 0 < args.fraction_estimate < 1, "fraction-estimate needs to be a float between 0 and 1"

# path for saving .png
networkPathLine = Path(mainPath) / "experiments" / "savedNetworks" / "lineValueAgentParameters_230817_1.npy"
networkPathBasic = Path(mainPath) / "experiments" / "savedNetworks" / "trainBasicValueAgent_withOpponent_dominoeAgent.npy"
savePath = Path(mainPath) / "docs" / "media"
device = "cuda" if torchCuda.is_available() else "cpu"

if __name__ == "__main__":
    numPlayers = args.num_players
    highestDominoe = args.highest_dominoe

    # create a league manager with the requested parameters
    league = lm.leagueManager(highestDominoe, numPlayers, shuffleAgents=True, device=device)

    # add copies of each agent type

    # Note: adding copies allow some matches to be played between agents of the same type,
    # which I think is important for achieving a balanced estimate of ELO scores. This is
    # important because the average dominoe score is dependent on the other agents that
    # are playing (e.g. it will be higher if an agent is good at going out fast). At the
    # end, this experiment averages the ELO for each agent type.
    numEach = args.num_each
    for _ in range(numEach):
        # instantiate lineValueAgent
        cAgent = da.lineValueAgent(numPlayers, highestDominoe, league.dominoes, league.numDominoes, device=league.device)
        cAgent.loadAgentParameters(networkPathLine)  # load parameters of a highly-trained agent
        cAgent.setLearning(False)  # I don't want these agents to update their parameters anymore
        league.addAgent(cAgent)  # add agent to league

    for _ in range(numEach):
        # instantiate basicValueAgent
        cAgent = da.basicValueAgent(numPlayers, highestDominoe, league.dominoes, league.numDominoes, device=league.device)
        cAgent.loadAgentParameters(networkPathBasic)
        cAgent.setLearning(False)
        league.addAgent(cAgent)

    # Then add all the other agent types
    league.addAgentType(da.persistentLineAgent, num2add=numEach)
    for agentIdx in range(2 * numEach, 3 * numEach):
        league.getAgent(agentIdx).maxLineLength = 12
    league.addAgentType(da.doubleAgent, num2add=numEach)
    league.addAgentType(da.greedyAgent, num2add=numEach)
    league.addAgentType(da.dominoeAgent, num2add=numEach)
    league.addAgentType(da.stupidAgent, num2add=numEach)

    assert numPlayers <= league.numAgents, "the number of players must be less than the number of agents in the league!"

    numGames = args.num_games
    num2EstimateWith = int(numGames * args.fraction_estimate)

    # Run lots of games, update and track ELO scores
    trackElo = np.zeros((numGames, league.numAgents))
    for gameIdx in tqdm(range(numGames)):
        game, leagueIndex = league.createGame()
        game.playGame()
        league.updateElo(leagueIndex, game.currentScore)  # update ELO
        trackElo[gameIdx] = copy(league.elo)

    # average ELO scores across agent type, and get agentType names
    numAgentTypes = 6
    avgEloPerAgentType = np.mean(trackElo.T.reshape(numAgentTypes, numEach, numGames), axis=1)
    agentTypeNames = [agent.agentName for agent in league.agents[::numEach]]

    # use last fraction of ELO estimates to get an averaged estimate of ELO
    eloEstimate = np.mean(avgEloPerAgentType[:, -num2EstimateWith:], axis=1)
    for name, elo in zip(agentTypeNames, eloEstimate):
        print(f"Agent {name} has a final ELO of {elo:.1f}")

    # Get these colors to coordinate across subplots
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors = [colors[i] for i in range(numAgentTypes)]

    # Plot ELO trajectory and final ELO estimates for each agent type
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (name, elo) in enumerate(zip(agentTypeNames, avgEloPerAgentType)):
        ax[0].plot(range(numGames), elo, label=name, linewidth=2, c=colors[idx])
    ax[0].set_xlabel("Number of games")
    ax[0].set_ylabel("Average ELO")
    ax[0].set_ylim(0, 2000)
    ax[0].legend(fontsize=12, loc="lower left")

    ax[1].bar(range(numAgentTypes), eloEstimate, color=colors, tick_label=agentTypeNames)
    plt.xticks(rotation=15)
    ax[1].set_ylabel("ELO")
    ax[1].set_ylim(0, 2000)
    if not (args.nosave):
        plt.savefig(str(savePath / "lineValueAgentELOs.png"))  # save figure and show to user
    plt.show()
