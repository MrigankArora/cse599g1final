Next Steps
==========

There’s definite room for improvement. I imagine changing the input of the graph from a simple vector to an embedding could help massively.
There's plenty of ways to format inputs for graphs, each of which contains slightly different information that the model could use in making its decision.
Depending on which scoring function you use, some input formats for graphs may be more useful than others. For instance, a scoring function based on vertex degrees for a
directed graph may benefit more from an adjacency matrix style input than a simple vector that just iterates through the edges.

Investigation into other Reinforcement Learning algorithms could be just as useful. The issue with the most popular RL algorithms is that for this problem,
scoring functions can only score the final state, and so there is no intermediate scoring function or heuristic the model can use. So the RL algorithm would have
to be able to learn from states with the knowledge of how it might do in the future.

The issue mentioned in the previous paragraph also lends itself to another extension, which is to develop heuristics for the scoring function. My idea is along the lines of a function approximator
that scores both the final states and the intermediate ones. This is definitely the hardest, but potentially most rewarding extension to work on.

Thanks for reading!
