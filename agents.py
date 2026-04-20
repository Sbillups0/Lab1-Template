import random
from model import (
    GoblinMoves,
    GameState,
    GameAction,
    WizardMoves,
    GameTransitions,
    Goblin,
    Wizard,
    EmptyEntity,
)
import sys


class EntityAgent:
    id: int

    def react(self, state: GameState) -> GameAction:
        raise NotImplementedError()


class WizardAgent(EntityAgent):
    def __init__(self, initial_state: GameState):
        pass

    def react(self, state: GameState) -> WizardMoves:
        # given new game state, update any internal state and return next move
        return WizardMoves.STAY


class WizardSearchAgent(WizardAgent):
    plan: list[WizardMoves] = []

    def __init__(self, initial_state: GameState):
        pass

    def react(self, state: GameState) -> WizardMoves:
        if self.plan:
            return self.plan.pop()
        else:
            self.start_search(state)
            return WizardMoves.STAY

    def start_search(self, game_state: GameState):
        pass

    def next_search_expansion(self) -> GameState | None:
        pass

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        pass


class ReasoningWizard(WizardAgent):
    nodes_expanded: int = 0
    max_depth: int = 1

    def get_successors(
        self, source: GameState
    ) -> tuple[tuple[GameAction, GameState], ...]:
        self.nodes_expanded += 1
        return GameTransitions.get_successors(source) # Get the successor states and actions from the game transitions

    def evaluation(self, state: GameState) -> float:
        return state.score

    def react(self, state: GameState) -> WizardMoves:
        values: dict[WizardMoves, float] = {} # Dictionary mapping actions to their evaluation scores (empty at start)
        for action, result in self.get_successors(state):
            values[action] = self.evaluation(result) #Assign eval score to each action (using our eval func)

        return max(values, key=values.get) # Return the action with the highest eval score (already greedy)


class GoblinAgent(EntityAgent):
    def react(self, state: GameState) -> GoblinMoves:
        # given new game state, update any internal state and return next move
        return GoblinMoves.STAY


class RandomGoblinAgent(GoblinAgent):
    def react(self, state: GameState) -> GoblinMoves:
        return random.choice(list(GoblinMoves))


class GreedyGoblinAgent(GoblinAgent):
    def react(self, state: GameState) -> GoblinMoves:
        distances: dict[GameAction, float] = {}
        for action, result in GameTransitions.get_successors(state):
            wizard_locs = [loc for loc in result.get_all_entity_locations(Wizard)]
            # Always kill the wizard if possible
            if not wizard_locs:
                return action
            else:
                wizard_loc = wizard_locs[0]
                goblin_locs = [
                    loc
                    for loc in result.get_all_entity_locations(Goblin)
                    if result.entity_grid[loc.row][loc.col].id == self.id
                ]
                if not goblin_locs:
                    raise RuntimeError(
                        f"No goblins in game state when calculating goblin moves! for goblin {self} in {state}"
                    )
                goblin_loc = goblin_locs[0]
                distances[action] = abs(wizard_loc.row - goblin_loc.row) + abs(
                    wizard_loc.col - goblin_loc.col
                )
        greedy_action = min(distances, key=distances.get)
        return greedy_action
