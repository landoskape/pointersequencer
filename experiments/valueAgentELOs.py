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
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.cuda as torchCuda

# dominoes package
from dominoes import leagueManager as lm
from dominoes import agents as da

device = "cuda" if torchCuda.is_available() else "cpu"

# can edit this for each machine it's being used on
resPath = Path(mainPath) / "experiments" / "savedResults"
prmsPath = Path(mainPath) / "experiments" / "savedParameters"
figsPath = Path(mainPath) / "docs" / "media"

for path in (resPath, prmsPath, figsPath):
    if not (path.exists()):
        path.mkdir()

# paths for loading previously trained agents
networkPath = Path(mainPath) / "experiments" / "savedNetworks"
trainedNetworks = [
    # basic  value agents
    {
        "path": networkPath / "trainValueAgent_basicValueAgent_against_dominoeAgent.npy",
        "name": "basicValueAgent < dominoeAgent",
        "agent": da.basicValueAgent,
        "type": 0,
    },
    {
        "path": networkPath / "trainValueAgent_basicValueAgent_against_doubleAgent.npy",
        "name": "basicValueAgent < doubleAgent",
        "agent": da.basicValueAgent,
        "type": 0,
    },
    # line value agents
    {
        "path": networkPath / "trainValueAgent_lineValueAgent_against_dominoeAgent.npy",
        "name": "lineValueAgent < dominoeAgent",
        "agent": da.lineValueAgent,
        "type": 1,
    },
    {
        "path": networkPath / "trainValueAgent_lineValueAgent_against_doubleAgent.npy",
        "name": "lineValueAgent < doubleAgent",
        "agent": da.lineValueAgent,
        "type": 1,
    },
]
numTrained = len(trainedNetworks)

handCraftedAgents = [
    {"name": "persistentLineAgent", "agent": da.persistentLineAgent},
    {"name": "doubleAgent", "agent": da.doubleAgent},
    {"name": "greedyAgent", "agent": da.greedyAgent},
    {"name": "dominoeAgent", "agent": da.dominoeAgent},
    {"name": "stupidAgent", "agent": da.stupidAgent},
]


def handleArguments():
    # input arguments
    parser = argparse.ArgumentParser(description="Run dominoes experiment.")
    parser.add_argument("-np", "--num-players", type=int, default=4, help="the number of players for each game")
    parser.add_argument("-hd", "--highest-dominoe", type=int, default=9, help="highest dominoe value in the set")
    parser.add_argument("-ng", "--num-games", type=int, default=4000, help="how many games to play to estimate ELO")
    # note: ELO is probability based, so increasing the number of rounds will usually exaggerate differences in ELO
    parser.add_argument("-nr", "--num-rounds", type=int, default=None, help="how many rounds to play for each game")
    parser.add_argument(
        "-ne", "--num-each", type=int, default=2, help="how many copies of each agent to use in the league"
    )  # helps get a better average of ELO scores
    parser.add_argument("-fe", "--fraction-estimate", type=float, default=0.25, help="final fraction of elo estimates to use")
    parser.add_argument("--nosave", default=False, action="store_true")
    parser.add_argument("--justplot", default=False, action="store_true")

    args = parser.parse_args()
    assert 0 < args.fraction_estimate < 1, "fraction-estimate needs to be a float between 0 and 1"
    return args


def getFileName():
    return "valueAgentELOs"


# creates agent list based on the agent information stored in trained networks
def createAgentList(league, numCopies):
    totalTrained = numTrained * numCopies
    agents = [None] * totalTrained
    names = [None] * totalTrained
    types = [None] * totalTrained
    for idx, tn in enumerate(trainedNetworks):
        for ii in range(2):
            cidx = 2 * idx + ii
            agents[cidx] = tn["agent"](league.numPlayers, league.highestDominoe, league.dominoes, league.numDominoes, device=league.device)
            agents[cidx].loadAgentParameters(tn["path"])
            agents[cidx].setLearning(False)
            names[cidx] = tn["name"]
            types[cidx] = tn["type"]

    return agents, names, types


def estimateELO(numGames, numRounds):
    # create a league manager with the requested parameters
    league = lm.leagueManager(args.highest_dominoe, args.num_players, shuffleAgents=True, device=device)
    agents, names, types = createAgentList(league, args.num_each)
    league.addAgents(agents)

    for hcAgent in handCraftedAgents:
        league.addAgentType(hcAgent["agent"], num2add=args.num_each)
        names += [hcAgent["name"]] * args.num_each

    print("Measuring ELO with the following agents: ")
    for idx, (name, agent) in enumerate(zip(names, league.agents)):
        print(f"Agent in league: {name} -- (agentType: {agent.agentName})")

    # Run lots of games, update and track ELO scores
    trackElo = np.zeros((numGames, league.numAgents))
    trackScore = np.full((numGames, league.numAgents), np.nan)
    trackHandWins = np.full((numGames, league.numAgents), np.nan)
    numRounds = numRounds if numRounds is not None else league.highestDominoe + 1
    for gameIdx in tqdm(range(numGames)):
        game, leagueIndex = league.createGame()
        game.playGame(rounds=numRounds)
        league.updateElo(leagueIndex, game.currentScore)  # update ELO
        trackElo[gameIdx] = copy(league.elo)
        trackScore[gameIdx, leagueIndex] = game.currentScore / numRounds  # track agent score (average per hand)
        trackHandWins[gameIdx, leagueIndex] = np.sum(game.score == 0, axis=0) / numRounds  # track how many times each agent won a hand

    # Estimate final ELO
    num2EstimateWith = int(numGames * args.fraction_estimate)
    eloEstimate = np.mean(trackElo[-num2EstimateWith:], axis=0)
    averageScore = np.nanmean(trackScore, axis=0)
    averageHandWins = np.nanmean(trackHandWins, axis=0)

    # Create results array
    results = {"elo": eloEstimate, "averageScore": averageScore, "averageHandWins": averageHandWins, "trackedElo": trackElo, "names": names}

    return results


# And a function for plotting results
def plotResults(results, args):
    elo = np.mean(results["elo"].reshape(-1, args.num_each), axis=1)
    averageScore = np.mean(results["averageScore"].reshape(-1, args.num_each), axis=1)
    averageHandWins = np.mean(results["averageHandWins"].reshape(-1, args.num_each), axis=1)
    trackedElo = np.mean(results["trackedElo"].T.reshape(-1, args.num_each, args.num_games), axis=1)

    names = results["names"][:: args.num_each]

    numAgents = len(names)

    # Show plot of tracked ELO trajectories to make sure it reached asymptotic ELO ratings
    f1 = plt.figure(1)
    for name, telo in zip(names, trackedElo):
        plt.plot(range(args.num_games), telo, label=name)
    plt.ylim(0)
    plt.legend(loc="best")
    plt.show()

    # Create discrete colormap
    cmap = mpl.colormaps["Dark2"]
    norm = mpl.colors.Normalize(vmin=0, vmax=numAgents - 1)
    colors = [cmap(norm(i)) for i in range(numAgents)]

    f2, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].bar(x=range(numAgents), height=elo, color=colors, tick_label=names)
    ax[0].tick_params(labelrotation=25)
    ax[0].set_ylim(0)
    ax[0].set_ylabel("ELO")

    ax[1].bar(x=range(numAgents), height=averageScore, color=colors, tick_label=names)
    ax[1].tick_params(labelrotation=25)
    ax[1].set_ylim(0)
    ax[1].set_ylabel("avg score/hand")

    ax[2].bar(x=range(numAgents), height=averageHandWins, color=colors, tick_label=names)
    ax[2].tick_params(labelrotation=25)
    ax[2].set_ylim(0)
    ax[2].set_ylabel("avg fraction of hand wins")

    if not (args.nosave):
        plt.savefig(str(figsPath / getFileName()))

    plt.show()


if __name__ == "__main__":
    args = handleArguments()

    if not (args.justplot):
        # estimate ELO with the requested parameters and agents
        results = estimateELO(args.num_games, args.num_rounds)

        # save results if requested
        if not (args.nosave):
            # Save agent parameters
            np.save(prmsPath / getFileName(), vars(args))
            np.save(resPath / getFileName(), results)

    else:
        prms = np.load(prmsPath / (getFileName() + ".npy"), allow_pickle=True).item()
        assert (
            prms.keys() <= vars(args).keys()
        ), f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(args).keys())}"
        for (pk, pi), (ak, ai) in zip(prms.items(), vars(args).items()):
            if pk == "justplot":
                continue
            if pk == "nosave":
                continue
            if prms[pk] != vars(args)[ak]:
                print(f"Requested argument {ak}={ai} differs from saved, which is: {pk}={pi}. Using saved...")
                setattr(args, pk, pi)

        results = np.load(resPath / (getFileName() + ".npy"), allow_pickle=True).item()

    plotResults(results, args)

    print(f"ELO: {results['elo']}")
    print(f"AvgScore: {results['averageScore']}")
    print(f"AvgHandWins: {results['averageHandWins']}")
