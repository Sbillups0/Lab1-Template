from model import (
    Location,
    Portal,
    Wizard,
    Wall,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import ReasoningWizard
from dataclasses import dataclass


class WizardGreedy(ReasoningWizard):
    def __init__(self, initial_state: GameState):
        # One time setup to get info needed for eval (portal locations)
        super().__init__(initial_state)
        self._portal_distances = self.compute_portal_distances(initial_state)
    
    def compute_portal_distances(self, state: GameState) -> dict[Location, float]:
        # BFS from portal to each reachable location to compute distance to portal for use in eval
        portal_locs = state.get_all_tile_locations(Portal)
        if not portal_locs:
            return {} # No portals, return empty dict
        
        portal_loc = portal_locs[0] # Assume only one portal for now
        distances = {portal_loc: 0} # Distance from portal to itself is 0
        queue = [portal_loc]
        head = 0
        while head < len(queue):
            loc = queue[head] # Get the next location to explore from the queue
            head += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Check all four cardinal directions
                next_loc = Location(loc.row + dr, loc.col + dc)
                in_bounds = (
                    0 <= next_loc.row < state.grid_size[0]
                    and 0 <= next_loc.col < state.grid_size[1]
                )
                if(
                    in_bounds
                    and next_loc not in distances # Not yet visited
                    and not isinstance(state.tile_grid[next_loc.row][next_loc.col], Wall) # Not a wall
                ):
                    distances[next_loc] = distances[loc] + 1
                    queue.append(next_loc)
        return distances
    
    def evaluation(self, state: GameState) -> float:
        wizard_locs = state.get_all_entity_locations(Wizard)

        if not wizard_locs:
            return -1000.0 # Wizard is dead, very bad state
        wizard_loc = wizard_locs[0]
        portal_locs = state.get_all_tile_locations(Portal)

        if portal_locs and isinstance(state.tile_grid[wizard_loc.row][wizard_loc.col], Portal):
            return 500.0 + state.score * 10.0 # Optimal win state, very good        
        total = state.score * 10.0 # Base score from game state (multiplied to have more impact relative to distance)

        #Use precomputer portal distances instead of Manhattan (had prior issues with walls) 
        portal_distance = self._portal_distances.get(wizard_loc, 9999) # Get distance to portal from precomputation (9999 if not reachable)
        total -= 2.0 * portal_distance # Closer to portal is better, so subtract distance from total score

        goblin_locs = state.get_all_entity_locations(Goblin)
        if goblin_locs:
            nearest_goblin_distance = min(
                abs(wizard_loc.row - goblin_loc.row) + abs(wizard_loc.col - goblin_loc.col)
                for goblin_loc in goblin_locs
            )
            total -= 30.0 / (nearest_goblin_distance + 1) # Closer goblins are worse, so subtract inverse of distance from total score (add 1 to avoid division by zero)
        return total

# BFS from portal to each reachable location to compute distance to portal for use in eval (tie breaker)
def bfs_portal_distances(state: GameState) -> dict:
        portal_locs = state.get_all_tile_locations(Portal)
        if not portal_locs:
            return {} # No portals, return empty dict
        
        portal_loc = portal_locs[0] # Assume only one portal for now
        distances = {portal_loc: 0} # Distance from portal to itself is 0
        queue = [portal_loc]
        head = 0

        while head < len(queue):
            loc = queue[head] # Get the next location to explore from the queue
            head += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Check all four cardinal directions
                next_loc = Location(loc.row + dr, loc.col + dc)
                in_bounds = (
                    0 <= next_loc.row < state.grid_size[0]
                    and 0 <= next_loc.col < state.grid_size[1]
                )
                if(
                    in_bounds
                    and next_loc not in distances # Not yet visited
                    and not isinstance(state.tile_grid[next_loc.row][next_loc.col], Wall) # Not a wall
                ):
                    distances[next_loc] = distances[loc] + 1
                    queue.append(next_loc)
        return distances

#Shared eval function 
def shared_evaluation(state: GameState, portal_distances: dict[Location, float]) -> float:
        wizard_locs = state.get_all_entity_locations(Wizard)
        if not wizard_locs:
            return -1000000 # Wizard is dead, very bad state, want to avoid at all costs
        wizard_loc = wizard_locs[0]
        #Winning sooner is better (so higher multiplier)
        if isinstance(state.tile_grid[wizard_loc.row][wizard_loc.col], Portal):
            # Optimal win state, very good, want to prioritize winning above all else, then within winning states prefer higher score from crystals
            return 1000000.0 + 100.0*state.score - 10.0*state.turn 
        total = 0.0
        #Reward for crystals, multiplied to have more impact relative to distance
        total += 80 * state.score 
        # strongly reward progress towards portal
        portal_distance = portal_distances.get(wizard_loc, 9999) 
        total -= 6.0 * portal_distance # Closer to portal is better, so subtract distance from total score

        #discourage stalling/loops (issue previously)
        total -= 2.0 * state.turn

        goblin_locs = state.get_all_entity_locations(Goblin)
        if goblin_locs:
            nearest_goblin_distance = min(
                abs(wizard_loc.row - goblin_loc.row) + abs(wizard_loc.col - goblin_loc.col)
                for goblin_loc in goblin_locs
            )
            if nearest_goblin_distance == 0:
                total -= 1000000.0 # Goblin on same tile is instant death, worst state, want to avoid at all costs
            elif nearest_goblin_distance == 1:
                total -= 500.0 # Adjacent goblin is very bad, about to die, want to avoid
            elif nearest_goblin_distance == 2:
                total -= 120.0 # Goblin two tiles away is somewhat dangerous, want to avoid if possible
            else:
                total -= 25.0 / nearest_goblin_distance # Further goblins are less dangerous, subtract inverse of distance from total score (add 1 to avoid division by zero)

        return total

#Shared terminal state function (remains same for all agents since terminal states are defined by the game rules, not the agent's strategy)
def is_terminal(state: GameState) -> bool:
        wizard_locs = state.get_all_entity_locations(Wizard)
        if not wizard_locs:
            return True
        wizard_loc = wizard_locs[0]
        portal_locs = state.get_all_tile_locations(Portal)
        return bool(portal_locs) and isinstance(state.tile_grid[wizard_loc.row][wizard_loc.col], Portal)  # Terminal if wizard on portal tile

#Get the successors of the state and order them by their evaluation score (lowest first if reverse=False, highest first if reverse=True) for use in alpha-beta pruning
def ordered_successors(agent: ReasoningWizard, state: GameState, reverse: bool):
    successors = list(agent.get_successors(state))
    successors.sort(key=lambda pair: agent.evaluation(pair[1]), reverse=reverse) # Order successors by evaluation score for better alpha-beta pruning performance
    return successors

class WizardMiniMax(ReasoningWizard):
    def __init__(self, initial_state: GameState):
        # One time setup to get info needed for eval (portal locations)
        super().__init__(initial_state)
        self._portal_distances = self.compute_portal_distances(initial_state)
    
    #Same BFS from portal as previous greedy agent to compute distances to portal for use in evaluation function
    def compute_portal_distances(self, state: GameState) -> dict[Location, float]:
        return bfs_portal_distances(state)
    
    max_depth: int = 2 # How many moves ahead the agent will look 

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        return shared_evaluation(state, self._portal_distances) # Use the shared evaluation function defined above with precomputed portal distances
        

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        return is_terminal(state) # Use the shared terminal state function defined above

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        best_action = WizardMoves.STAY # Default action if no successors
        best_value = float('-inf') # Initialize best value to negative infinity for maximization

        for action, successor in ordered_successors(self, state, reverse=True): # Get the successor states ordered by their evaluation score (lowest first) for better alpha-beta pruning performance
            value = self.minimax(successor, self.max_depth - 1) # Get minimax value of successor state (subtract 1 from depth since we are going down one level in the tree)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action


    def minimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        if self.is_terminal(state):
            return self.evaluation(state)
        
        active = state.get_active_entity() #Determines wizard/goblin turn
        if isinstance(active, Wizard): #Wizard turn, maximizing for wizard
            if depth == 0:
                return self.evaluation(state)
            best = float('-inf') # Maximizing for wizard
            for action, successor in ordered_successors(self, state, reverse=True): # Get the successor states ordered by their evaluation score (highest first) for better alpha-beta pruning performance
                best = max(best, self.minimax(successor, depth - 1)) # Get the max value of the successor states
            return best
        else: # Goblin turn, minimizing for goblin
            worst = float('inf')
            for action, successor in ordered_successors(self, state, reverse=False): # Get the successor states ordered by their evaluation score (lowest first) for better alpha-beta pruning performance
                worst = min(worst, self.minimax(successor, depth)) # Get the min value of the successor states (keeping depth the same since we only want to decrement on wizard turns)
            return worst


class WizardAlphaBeta(ReasoningWizard):
    #Similar to minimax, but with alpha-beta pruning to avoid expanding branches that won't impact the final decision, allowing for deeper search within the same time limit

    max_depth: int = 2
    def __init__(self, initial_state: GameState):
        # One time setup to get info needed for eval (portal locations)
        super().__init__(initial_state)
        self._portal_distances = self.compute_portal_distances(initial_state)
    def compute_portal_distances(self, state: GameState) -> dict[Location, float]:
        return bfs_portal_distances(state)
    
    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        return shared_evaluation(state, self._portal_distances) # Use the shared evaluation function defined above with precomputed portal distances
    
    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        return is_terminal(state) # Use the shared terminal state function defined above

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        best_action = WizardMoves.STAY # Default action if no successors
        best_value = float('-inf') # Initialize best value to negative infinity for maximization
        alpha = float('-inf') # Initialize alpha to negative infinity
        beta = float('inf') # Initialize beta to positive infinity

        for action, successor in ordered_successors(self, state, reverse=True): # Get the successor states ordered by their evaluation score (highest first)
            value = self.alpha_beta_minimax(successor, self.max_depth - 1, alpha, beta) # Get alpha-beta minimax value of successor state (subtract 1 from depth since we are going down one level in the tree)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value) # Update alpha with the best value found so far

        return best_action

    def alpha_beta_minimax(self, state: GameState, depth: int, alpha: float, beta: float):
        # TODO YOUR CODE HERE
        if self.is_terminal(state):
            return self.evaluation(state)
        
        active = state.get_active_entity() #Determines wizard/goblin turn
        if isinstance(active, Wizard): #MAX node (wizard turn)
            if depth == 0:
                return self.evaluation(state)
            
            best = float('-inf')
            for action, successor in ordered_successors(self, state, reverse=True): # Get the successor states ordered by their evaluation score (highest first) 
                best = max(best, self.alpha_beta_minimax(successor, depth - 1, alpha, beta)) # Get the max value of the successor states
                alpha = max(alpha, best) # Update alpha with the best value found so far
                if alpha >= beta:
                    break # Beta cut-off, no need to explore further since the minimizing player would never allow this branch to be chosen
            return best
        else: # MIN node (goblin turn)
            worst = float('inf')
            for action, successor in ordered_successors(self, state, reverse=False): # Get the successor states ordered by their evaluation score (lowest first)
                worst = min(worst, self.alpha_beta_minimax(successor, depth, alpha, beta)) # Get the min value of the successor states (keeping depth the same since we only want to decrement on wizard turns)
                beta = min(beta, worst) # Update beta with the worst value found so far
                if alpha >= beta:
                    break # Alpha cut-off, no need to explore further since the maximizing player would never allow this branch to be chosen
            return worst



class WizardExpectimax(ReasoningWizard):
    max_depth: int = 2
    def __init__(self, initial_state: GameState):
        # One time setup to get info needed for eval (portal locations)
        super().__init__(initial_state)
        self._portal_distances = self.compute_portal_distances(initial_state)
    def compute_portal_distances(self, state: GameState) -> dict[Location, float]:
        return bfs_portal_distances(state)
    
    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        wizard_locs = state.get_all_entity_locations(Wizard)
        if not wizard_locs:
            return -1000.0 # Wizard is dead, very bad state
        wizard_loc = wizard_locs[0]

        if isinstance(state.tile_grid[wizard_loc.row][wizard_loc.col], Portal):
            return 500.0 + state.score * 10.0 # Optimal win state, very good
        total = state.score * 10.0 # Base score from game state (multiplied to have more impact relative to distance)
        portal_distance = self._portal_distances.get(wizard_loc, 9999) # Get distance to portal from precomputation (9999 if not reachable)
        total -= 1.5 * portal_distance # Closer to portal is better

        goblin_locs = state.get_all_entity_locations(Goblin)
        if goblin_locs:
            nearest_goblin_distance = min(
                abs(wizard_loc.row - goblin_loc.row) + abs(wizard_loc.col - goblin_loc.col)
                for goblin_loc in goblin_locs
            )
            
            #Note: reduced penalties for goblins compared to minimax agent since expectimax assumes random behavior and is less certain about the threat, 
            # we want to be less aggressive about avoiding them to allow for more exploration and potentially better long-term outcomes
            if nearest_goblin_distance == 0:
                total -= 1000.0 # Goblin on same tile is instant death, worst state
            elif nearest_goblin_distance == 1:
                total -= 300.0 # Adjacent goblin is very bad, about to die
            elif nearest_goblin_distance == 2:
                total -= 40.0 # Goblin two tiles away is somewhat dangerous
            
        #Crystal incentive (encourage collecting crystals for scoring if nearby)
        crystal_locs = state.get_all_entity_locations(Crystal)
        for c in crystal_locs:
            distance = abs(wizard_loc.row - c.row) + abs(wizard_loc.col - c.col)
            if distance <= 2:
                total += 6.0 / (distance + 1) # Closer crystals are more valuable, add inverse of distance to score (add 1 to avoid division by zero)
        
        return total

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        return is_terminal(state) # Use the shared terminal state function defined above

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        best_action = WizardMoves.STAY # Default action if no successors
        best_value = float('-inf') # Initialize best value to negative infinity for maximization
        best_dist = float('inf') # For tie-breaking, prefer states closer to portal

        for action, successor in self.get_successors(state):
            value = self.expectimax(successor, self.max_depth - 1) # Get expectimax value of successor state (subtract 1 from depth since we are going down one level in the tree)
            wlocs = successor.get_all_entity_locations(Wizard)
            succ_dist = self._portal_distances.get(wlocs[0], 9999) if wlocs else 9999 # Get distance to portal in successor state for tie-breaking
            if value > best_value or (value == best_value and succ_dist < best_dist):
                best_value = value
                best_action = action
                best_dist = succ_dist
        return best_action


    def expectimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        if self.is_terminal(state):
            return self.evaluation(state)
        active = state.get_active_entity() #Determines wizard/goblin turn
        if isinstance(active, Wizard): #MAX node
            if depth == 0:
                return self.evaluation(state)
            best = float('-inf')
            for action, successor in self.get_successors(state):
                best = max(best, self.expectimax(successor, depth - 1)) # Get the max value of the successor states
            return best
        else: # Chance node (goblin turn, modelled as random)
            successors = self.get_successors(state)
            if not successors:
                return self.evaluation(state) # No successors, evaluate current state
            total = sum(self.expectimax(successor, depth) for action, successor in successors) # Get the sum of the values of the successor states
            return total / len(successors) # Return the average value of the successor states (expected value for random goblin behavior)
