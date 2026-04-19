from model import (
    Location,
    Portal,
    EmptyEntity,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import WizardSearchAgent
import heapq
from dataclasses import dataclass


class WizardDFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {} #Dictionary storing visited search states and the path of actions taken to reach them (from the initial state)
        self.paths[initial_search_state] = [] #Path to initial search state is empty since we start there
        self.search_stack = [initial_search_state] #Stack of search states to expand (frontier), initialized with the initial search state (i.e. considered paths, but not yet visited)

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc
    
    #Check if anything left to expand towards goal, if not return None, otherwise return next game state to expand towards goal
    def next_search_expansion(self) -> GameState | None:
        # TODO: YOUR CODE HERE
        if not self.search_stack:
            return None # No more nodes to expand (frontier empty), search failed
        
        state = self.search_stack.pop() # Get the next node to expand (from the top of the stack)
        return self.search_to_game(state) # Convert the search state to a game state and return it for expansion

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO: YOUR CODE HERE
        source_search_state = self.game_to_search(source) # Convert the source game state to a search state
        target_search_state = self.game_to_search(target) # Convert the target game state to a search state

        # Only process target if not yet visited (not in self.paths)
        if target_search_state not in self.paths:
            # Add path from source to target (path to source + action taken)
            self.paths[target_search_state] = self.paths[source_search_state] + [action]
            if self.is_goal(target_search_state):
                # Goal found, no need to add to search stack, but need to reverse path (so pop works correctly in returning actions) and save to self.plan
                self.plan = list(reversed(self.paths[target_search_state]))
            else:
                # Not goal, add to search stack for future expansion
                self.search_stack.append(target_search_state)


class WizardBFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        # TODO: YOUR CODE HERE
        if not self.search_stack:
            return None # No more nodes to expand (frontier empty), search failed
        
        state = self.search_stack.pop(0) # Get the next node to expand (from the front of the stack for BFS; FIFO)
        return self.search_to_game(state) # Convert the search state to a game state and return it for expansion

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO: YOUR CODE HERE
        source_search_state = self.game_to_search(source) # Convert the source game for comparison
        target_search_state = self.game_to_search(target) # Convert the target game for comparison
        # Only process target if not yet visited (not in self.paths)
        if target_search_state not in self.paths:
            # Add path from source to target (path to source + action taken)
            self.paths[target_search_state] = self.paths[source_search_state] + [action]
            if self.is_goal(target_search_state):
                # Goal found, no need to add to search stack, but need to reverse path (so pop works correctly in returning actions) and save to self.plan
                self.plan = list(reversed(self.paths[target_search_state]))
            else:
                # Not goal, add to search stack for future expansion
                self.search_stack.append(target_search_state)

class WizardAstar(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def heuristic(self, target: GameState) -> float:
        # TODO: YOUR CODE HERE
        target_ss = self.game_to_search(target) # Convert to SS for calculation
        wizard_loc = target_ss.wizard_loc 
        portal_loc = target_ss.portal_loc
        return abs(wizard_loc.row - portal_loc.row) + abs(wizard_loc.col - portal_loc.col) # Manhattan distance heuristic

    def next_search_expansion(self) -> GameState | None:
        # TODO: YOUR CODE HERE
        while self.search_pq: # While there are still nodes to expand (frontier not empty)
            f_cost, state = heapq.heappop(self.search_pq) # Get the next node to expand (lowest f-cost from the priority queue)(f = g + h)

            g_current, _ = self.paths[state] # Get the current g-cost for this state from self.paths 
            h = self.heuristic(self.search_to_game(state)) # Calculate the heuristic for this state
            
            # If the f-cost is greater than the g-cost + h, then we have already found a better path to this state, so we can skip it.
            # happens because we may have added this state to the priority queue multiple times with different f-costs as we 
            # found different paths to it, but we only want to expand the one with the lowest f-cost (the best path to it).
            if f_cost > g_current + h:
                continue
            if self.is_goal(state):
                _, path = self.paths[state] # Get the path to this state from self.paths and save to self.plan
                self.plan = list(reversed(path)) # Reverse the path so that we can pop actions from it in the correct order
                return None # Goal found, return None to indicate search is complete
            
            return self.search_to_game(state)
        return None

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO: YOUR CODE HERE
        source_ss = self.game_to_search(source) 
        target_ss = self.game_to_search(target) 

        g_source, path_source = self.paths[source_ss] # Get the g-cost and path to the source state from self.paths
        new_g = g_source + self.cost(source, target, action) # Calculate the new g-cost to the target state through this expansion (g-source + cost of action)
        new_f = new_g + self.heuristic(target) # Calculate the new f-cost to the target state through this expansion (new g + heuristic of target)

        # If the target state has not been visited yet, or we found a cheaper path to it (new g less than previously recorded g in self.paths)
        if target_ss not in self.paths or new_g < self.paths[target_ss][0]: 
            self.paths[target_ss] = new_g, path_source + [action]
            heapq.heappush(self.search_pq, (new_f, target_ss)) # Add the target state to the priority queue with its f-cost


class CrystalSearchWizard(WizardSearchAgent):
    # TODO: YOUR CODE HERE
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location
        remaining_crystals: tuple[Location, ...] # Need to track remaining crystals in search state for optimal crystal search
    
    initial_game_state: GameState

    def manhattan_distance(self, loc1: Location, loc2: Location) -> float:
        return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)
    
    def MST_cost(self, locations: list[Location]) -> float:
        # Prim's algorithm for MST cost
        if len(locations) <= 1:
            return 0
        in_mst = {locations[0]} # Start with the first location in the MST
        remaining = list(locations[1:]) # Remaining locations to add to the MST
        total_cost = 0
        while remaining:
            # Finds minimum cost edge from in_mst to remaining (i.e. minimum manhattan distance from any location in the MST to any location not yet in the MST)
            best_cost, best_loc = min(
                (self.manhattan_distance(loc, mst_loc), loc)
                for loc in remaining
                for mst_loc in in_mst
            )
            total_cost += best_cost
            in_mst.add(best_loc) # Add the best location to the MST
            remaining.remove(best_loc) # Remove the best location from the remaining list
        return total_cost
    
    def heuristic(self, target: GameState) -> float:
        if not target.remaining_crystals:
            return self.manhattan_distance(target.wizard_loc, target.portal_loc) # If no remaining crystals, heuristic is just the manhattan distance from the wizard to the portal
        
        # Prevent O(grid size) search issue annoyance by caching MST cost
        # MST(remaining + portal) is identical for all states with the same remaining set,
        # regardless of where the wizard is. Only the min-to-crystal leg changes per state.

        cache_key = target.remaining_crystals
        if cache_key not in self.mst_cache:
            crystal_portal_locs = list(target.remaining_crystals) + [target.portal_loc]
            self.mst_cache[cache_key] = self.MST_cost(crystal_portal_locs)

        min_to_crystal = min(self.manhattan_distance(target.wizard_loc, crystal_loc) for crystal_loc in target.remaining_crystals)

        return min_to_crystal + self.mst_cache[cache_key]
    
    def search_to_game(self, search_state: SearchState) -> GameState:
        cache_key = search_state.remaining_crystals
        if cache_key not in self.game_state_cache:
            # Construct a base game state for this crystal configuration (with wizard at portal and all crystals in their locations, but no active entity)
            initial_wizard_loc = self.initial_game_state.active_entity_location
            base = self.initial_game_state.replace_entity(initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity())
            remaining_set = set(search_state.remaining_crystals)
            for crystal_loc in self.initial_crystal_locs:
                if crystal_loc not in remaining_set:
                    base = base.replace_entity(crystal_loc.row, crystal_loc.col, EmptyEntity())
            self.game_state_cache[cache_key] = base
        
        #Place the wizard in the search state location on the cached base game state for this crystal configuration
        base = self.game_state_cache[cache_key]
        initial_wizard = self.initial_game_state.get_active_entity()
        return (
            base
            .replace_entity(search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard)
            .replace_active_entity_location(search_state.wizard_loc)
        )
    
    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        remaining_crystals = tuple(sorted(game_state.get_all_entity_locations(Crystal)))
        return self.SearchState(wizard_loc, portal_loc, remaining_crystals)
    
    # Added to fix O(grid size) search issue
    def target_to_search(self, source_ss: SearchState, target: GameState) -> SearchState:
        new_wizard_loc = target.active_entity_location
        new_remaining = tuple(loc for loc in source_ss.remaining_crystals if loc != new_wizard_loc) # If we moved onto a crystal, it is no longer remaining, so remove it from the remaining crystals in the new search state
        return self.SearchState(new_wizard_loc, source_ss.portal_loc, new_remaining)


    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def is_goal(self, state: SearchState) -> bool:
        return len(state.remaining_crystals) == 0 and state.wizard_loc == state.portal_loc

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state
        self.portal_loc = game_state.get_all_tile_locations(Portal)[0]
        self.initial_crystal_locs = game_state.get_all_entity_locations(Crystal)
        self.visited = set() # Set of visited search states to avoid re-expanding the same state multiple times
        #Pointers to avoid storing full paths in self.paths for every state (caused issues with crystal map)
        self.mst_cache = {}  # Cache MST(remaining_crystals + portal) by remaining_crystals key
        self.game_state_cache = {}  # Cache wizard-free base states by crystal configuration (avoid reconstructing every time)
        self.parent = {} # Dictionary mapping search states to their parent search state (the state from which they were expanded)
        self.g_costs = {} 
        
        initial_state = self.game_to_search(game_state)
        self.g_costs[initial_state] = 0 # Cost to reach the initial state is 0
        self.search_pq = [(self.heuristic(initial_state), initial_state)] # Priority queue of search states to expand (frontier), initialized with the initial state and its heuristic value as the priority
        self.current_search_state = initial_state # Track the current search state for use in target_to_search to avoid O(grid size) search issue

    def reconstruct_path(self, goal_state: SearchState) -> list[WizardMoves]:
        path = []
        state = goal_state
        while state in self.parent: #trace pointers back to start state to reconstruct path (until we reach the initial state which has no parent)
            parent_state, action = self.parent[state]
            path.append(action)
            state = parent_state
        return path

    def next_search_expansion(self) -> GameState | None:
        # TODO YOUR CODE HEREs
        while self.search_pq: # While there are still nodes to expand (frontier not empty)
            _, state = heapq.heappop(self.search_pq) # Get the next lowest cost target to expand from pq
            if state in self.visited:
                continue # If we have already visited this state, skip it, continuing until unvisted state or empty pq
            if self.is_goal(state):
                self.plan = self.reconstruct_path(state) # If goal, reconstruct path to it using pointers and save to self.plan
                return None # Goal found, return None to indicate search is complete
            
            self.visited.add(state) # Mark this state as visited
            self.current_search_state = state # Update the current search state for use in target_to_search to avoid O(grid size) search issue
            return self.search_to_game(state) # return for expansion
        
        return None # IF pq is empty, no more nodes to expand and search failed

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO YOUR CODE HERE
        source_ss = self.current_search_state
        target_ss = self.target_to_search(source_ss, target) # Convert the target game state to a search state (using target_to_search to avoid O(grid size) search issue)

        if target_ss in self.visited:
            return # If we have already visited the target state, do not process it
        
        # Otherwise, calculate the new path to the target state through this expansion (path to source + action taken)
        new_cost = self.g_costs[source_ss] + 1 # Since cost in this case refers to path length to portal, new cost is just source cost + 1

        if target_ss not in self.g_costs or new_cost < self.g_costs[target_ss]: # If target state not yet visited or we found a cheaper path to it (new cost less than previously recorded cost in self.paths)
            self.g_costs[target_ss] = new_cost
            self.parent[target_ss] = (source_ss, action) # Update the parent pointer (no more list copying for path)
            priority = new_cost + self.heuristic(target_ss) 
            heapq.heappush(self.search_pq, (priority, target_ss)) # Add the target state to the priority queue

class SuboptimalCrystalSearchWizard(CrystalSearchWizard):
    # Need to add more to Search State 
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location
    def heuristic(self, target: SearchState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError
