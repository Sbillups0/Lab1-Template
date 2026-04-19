from agents import (
    RandomGoblinAgent,
    GoblinAgent,
    GreedyGoblinAgent,
    ReasoningWizard,
)
from model import Goblin, Wizard
import pyglet
from part1Agents import (
    WizardDFS,
    WizardBFS,
    WizardAstar,
    CrystalSearchWizard, SuboptimalCrystalSearchWizard,
)
from part2Agents import (
    WizardMiniMax,
    WizardExpectimax,
    WizardAlphaBeta,
    WizardGreedy,
)


from game import SearchGame
import argparse


parser = argparse.ArgumentParser()


parser.add_argument(
    "--agent",
    type=str,
    default="dfs",
    help="Search Agents: dfs, bfs, ucs, astar, crystal, suboptimal, greedy, minimax, expectimax, alphabeta",
)

parser.add_argument(
    "--goblin",
    type=str,
    default="lazy",
    help="Goblin Agents: lazy, random, greedy",
)

parser.add_argument("--map", type=str, default="dungeon", help="Map name")


parser.add_argument("--speed", type=float, default=10, help="Moves per second")
parser.add_argument(
    "--depth", type=int, default=4, help="Maximum search depth for reasoning agents"
)

parser.add_argument(
    "--timeout", type=int, default=60, help="Maximum time (seconds) to run"
)

parser.add_argument(
    "--search_only", action="store_true", help="Whether to run game after search"
)

parser.add_argument(
    "--hide_search", action="store_true", help="Whether to not render search nodes"
)
parser.add_argument(
    "--no_render", action="store_true", help="Whether to not render search nodes"
)
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

available_agents = {
    "dfs": WizardDFS,
    "bfs": WizardBFS,
    "astar": WizardAstar,
    "crystal": CrystalSearchWizard,
    "suboptimal": SuboptimalCrystalSearchWizard,
    "greedy": WizardGreedy,
    "minimax": WizardMiniMax,
    "expectimax": WizardExpectimax,
    "alphabeta": WizardAlphaBeta,
}

available_goblins = {
    "lazy": GoblinAgent,
    "random": RandomGoblinAgent,
    "greedy": GreedyGoblinAgent,
}


if __name__ == "__main__":
    game = SearchGame(
        path=f"maps/{args.map}",
        game_tick_interval=1.0 / args.speed,
        render_search=not args.hide_search,
        no_render=args.no_render,
        debug=args.debug,
        timeout=args.timeout,
    )

    for _ in game.state.get_all_entity_locations(Wizard):
        agent = available_agents[args.agent](game.state)
        if isinstance(agent, CrystalSearchWizard):
            game.require_crystal = True
        elif isinstance(agent, ReasoningWizard):
            agent.max_depth = args.depth

        game.register_next_wizard_agent(agent)

    for _ in game.state.get_all_entity_locations(Goblin):
        game.register_next_goblin_agent(available_goblins[args.goblin]())

    game.run()

    if not args.no_render:
        pyglet.app.run()
