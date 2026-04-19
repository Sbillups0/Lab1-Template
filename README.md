# Lab 1: Dungeon Crawler

In this lab you will program agents to search paths through a dungeon world.


## Files you will edit
>`part1Agents.py` : This will include all of your code for the search algorithms in part 1


>`part2Agents.py` : This will include all of your code for adversarial search algorithms in part 2

## Files you might want to read
>`model.py` : This includes all of the data model for the lab, including the classes representing game state, actions, and transitions. 

> `agents.py` : This includes all of the abstract agent classes that your agents will inherit from, as well as the code for the various adversarial goblin agents you may find

> `run.py` : This is the file to run to test your agents locally, you can run your agents against any of our maps (or make your own map to test) including a number of options you may find useful in testing.

## Submission
You will fill in portions of `part1Agents.py` and `part2Agents.py`  during the assignment. You can test your solutions to see if they are working how you expect using `run.py`. Once you have completed the assignment you will submit your files to gradescope (we will only look at `part1Agents.py` and `part2Agents.py` so do not edit any of the other files in your solution). You code will be autograded for technical correctness using test maps not in the repository, and there will be a few leader boards to see how efficient some of your design choices are. Please do not change the names of any provided functions or classes within the code, or import any additional packages, or you will wreak havoc on the autograder. Additionally, be careful to make your code as memory and time effeicient as possible, as the autograder machine has limited memory and if it fails or times out it will not be able to generate a full report. However, *remember that the actual correctness of your implementation alongside your lab review explanations will be the final judge of your score, rather than the autograder’s judgments*. 

One additional note, the autograder can sometimes take a long time to run due to the limited resources of the gradescope machine as well as running many trials in order to generate a reliable leaderboard score. Therefore you are *highly encouraged* to develop your own local testing to make sure your code is working the way you think instead of relying on the autograder entirely.

## Academic Dishonesty
We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else’s code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don’t try. We trust you all to submit your own work only; please don’t let us down. If you do, we will pursue the strongest consequences available to us.

## Getting Help
You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, lab, and the discussion forum are there for your support; please use them. If you can’t make office hours, let us know and we will schedule more. We want these labs to be rewarding and instructional, not frustrating and demoralizing. *But, we don’t know when or how to help unless you ask.*

----

# Welcome to the Dungeon Dimension
You have programmed a robot wizard to enter into the dungeon dimension, a world full of twisting corridors, shiny crystals, and scary goblins.
Like the Golem, it will be your code that breaths life to the robot and determines how it will act.
Navigating this world and finding the portal back home will be your robot's first job in mastering it's domain. 

After cloning this repo you should be able to see how your robot wizard will act using different agent algorithms and on different maps using `run.py`. Make sure to check out the different options that will make visualizing and understanding how your agent works much easier. Note that without any implemented agents, your wizard just stands still and does nothing. 

# Part 1: A New Path

In `part1Agents.py` you will find the template code for a few different kinds of agent programs you will deploy to your robot. These agents all work by searching through the possible actions and states in the dungeon dimension, and forming a plan of actions to reach their goals.

## Depth First Search

Your first task is to produce an agent class that performs Depth First Search. In order to help you understand how this game world and search capabilities work we have partially filled in some of the class. 

In order to perform a search a `WizardSearchAgent` has the ability to choose game states to expand and learn their successors (this is the search transition function) as well as must have a way to process each new state that is found through a node expansion. After finding a path of actions which will lead to its goal, a search agent has a special `plan` attribute which is a list of `WizardMoves` defining the agent's found path. Remember that these need to be legal moves found through querying valid successors (valid directions, no moving through walls).

The way that a `WizardSearchAgent` is processed in the environment is that if the agent does not have a live `plan` the game will request a `next_search_expansion` and then pass all of the successors of the returned node to  `process_search_expansion` and repeat this until a `plan` is determined or no `next_search_expansion` is returned. It is up to the agent to use this information to maintain their search.

In particular note that the agent maintains a separate `SearchState` that is used to perform the search which only stores the relevant information for the search instead of an entire `GameState`, but in order to request successors from the game the search state needs to be converted to and from a `GameState`. Make sure to avoid expanding any already visited states.

Finally to check that the search is doing what you expect, you can visualize the search and node expansions using `python run.py --agent dfs --map dungeon`. 

## Breadth First Search
This is very similar to depth first search. Again write an agent that performs the search and check, does it find the optimal (i.e. shortest path) solution?


## A*
While BFS will find a fewest-actions path to the goal, it does so by expanding many more nodes than it generally needs to. Implement A* search wizard and determine an admissible (and hopefully consistent if you are using graph search!) heuristic and see how many fewer nodes need to be expanded. The imported `heapq` default library for priority queues will likely come in handy.

## Crystal Search
The real power of A* is only be apparent with a more challenging search problem. Now, it’s time to formulate a new problem and design a heuristic for it.

In some of the dungeons there are crystals that you want your wizard to collect and bring back through the portal.

Implement `CrystalSearchWizard` to search for the optimal path that collects all of the crystals in the map and exits the portal.

Determine an appropriate `SearchState` for the problem and design a non-trivial heuristic. Remember to make sure your heuristic is *admissible*. You should evaluate your agent on the map `many-crystals` which will stress test your solution similarly to the autograder. You should be able to achieve a search which finds the optimal solution in fewer than 40,000 search nodes, with the best solutions achieving under 30,000. Our solution requires 21740 search expansions (which we have no doubt can also be beat).

Your method will also generate a score on the Crystal Search leaderboard on gradescope where you can enter an anonomys leaderboard name and strive to create the best crystal finding robot. 

## Suboptimal Crystal Search

Finally, explore how efficient you could become if you loosen the requirement on finding the optimal solution. Design an inadmissible heuristic for `SuboptimalCrystalSearchWizard`. See how many fewer nodes you can explore, and at what cost to optimality. How close to admissible is your heuristic? What are the bounds relative to the optimal path for your agent?

# Part 2: The Goblins Strike Back

After finding the riches of the dungeon dimension, you have now discovered you are not the only agent down there. There are also Goblins that can destroy your robot wizard at a single touch. In this part you will design agents that can navigate the dungeons knowing that there are adversaries out to get you. 

In `part2Agents.py` you will find the template code for the agent programs you will deploy to your robot. These agents work by defining some evaluation function for preferable states, and then doing some kind of internal reasoning at each turn to determine the action it thinks will lead to the best outcome.


## Greedy Wizard

In `WizardGreedy` implement an `evaluation` function where the wizard will simply choose the action at every turn with the highest resultant evaluation. Check how your agent performs against different kinds of `Goblins` with: 
- `python run.py --agent greedy --goblin random --map arena`
- `python run.py --agent greedy --goblin greedy --map arena`

Your greedy wizard should be able to get to the portal more than half of the time against the random goblin.


## Minimax

In `WizardMiniMax` implement an adversarial agent that implements a slightly more general form of minimax. Some dungeons (like `bigmap`) have more than one goblin. Therefore you need to be able to support reasoning using MiniMax with more than one minimizing node per turn. 

In `WizardMiniMax` you should implement this generalized MiniMax as well as an evaluation function. In order to expand the minimax search tree, unlike in part 1 `ReasoningAgents` get direct access to calculating `GameState` successors using the `self.get_successors` function. 
 
 ### Additional Notes
- Your agent must adhere to `max_depth`, where the depth is based on *the number of Wizard moves rather than the depth of successors in the tree*. 
- You should probably implement the algorithm recursively using helper function(s).
- The correct implementation of minimax will lead to your wizard losing the game in some cases. Random and Greedy goblins are of course not optimal minimax agents, and so modeling them with minimax search may sometimes make mistakes. This is ok and expected, and your evaluation function can be designed with this in mind. 
- We will be checking your code to determine whether it explores the correct number of game states. This is the only reliable way to detect some very subtle bugs in implementations of minimax. As a result, the autograder will be very picky about how many times you call `self.get_successors`. If you call it any more or less than necessary, the autograder will complain. 

## Alpha-Beta Pruning

Make a new agent in `WizardAlphaBeta` that uses alpha-beta pruning to perform a more efficient minimax. You should see a speed-up (perhaps depth 3 alpha-beta will run as fast as depth 2 minimax). Once again we will be very careful about checking the number of times you call `self.get_successors` so make sure you use all of the discussed optimizations for an effective alpha-beta pruning including:

- Good ordering of branches based on your evaluation function
- Breaking on an inclusive comparison for alpha and beta values rather than exclusive 

## Improving your evaluation function

Based on your correct implementation of alpha-beta search, and using your evaluation function to differentiate, your wizard will be put through a trial of dungeons with both random and greedy goblins in multiple different maps (including `pacman` as well as other hidden autograder maps) where your agent will be scored on how well it is able to both reliably survive and get to the portal as well as collect crystals along the way. 

You will be rewarded 10 points for each crystal your wizard can bring back through the portal, but lose 50 points if your wizard is killed or it takes too long either processing or running around in circles (it isnt free to keep a portal open forever!). This score averages over mulitple trials will form another leaderboard where you can compete to design the best evaluation function to power alpha-beta and have the best adversarial wizard.

When designing your evaluation function consider that goblins may be acting randomly, and so may be hard to predict in mini-max. Consider what kinds of features to model (i.e. how you will expect to score, what is your proximity to danger, how likely you are to timeout before getting to the portal, etc...) as well as how to weight them.


## Expectimax

Finally, while minimax and alpha-beta are very powerful, they are sometimes too conservative when playing against non-optimal opponents like `RandomGoblin`. In this case, we might expect that modeling opponent behaviour using expectimax might perform better. In `WizardExpectimax`, implement an expectimax search and see how it performs against both `random` and `greedy` goblins.
