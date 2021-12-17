Methodology
================

Algorithm Summary
------------------------
The algorithm described below is often referred to the Deep Cross-Entropy algorithm

.. figure:: images/algo.png
    :width: 800

    Overview of the algorithm

The algorithm begins with initializing a Neural Network. We then use this Neural Network to construct
some graphs. These graphs are then scored using our reward function, **f**\ , and a top percentile of them
are chosen. The chosen "good" graphs are what we use to train the Neural Network. Once training is complete,
we repeat the cycle by generating new graphs again, scoring and filtering them, and then using what's left to train
the Network again.

This cycle repeats until a counter example that breaks the score bound is found, or the user stops it. We call each loop of this cycle a single iteration.
Once a counterexample graph is constructed by the neural network, the loop ends, and a visualization, and any other relevant information of the graph is outputted.

Input Format
------------
A graph on N nodes can be encoded by a vector with N(N-1)/2 elements, where each element is a 1 if the corresponding edge is in the graph,
and a 0 otherwise. However, since we want to input to the model a game 'state' for the neural network, this vector isn't enough. We also need to know which edges the model
has already considered.

This leads to a vector with N(N-1) elements. The first N(N-1)/2 elements are as described in the previous paragraph, a state of the graph as it is currently. The second half of the vector
is a one-hot encoding that represents which edge the graph is meant to consider. For instance, if the (i + N(N-1)/2)th element is 1, then the first i-1 edges of the graph have already been considered
and the model is meant to output whether the ith edge should be added to the graph or not.

Initializing the Neural Network
--------------------------------
The Neural Network itself takes in a graph state, and outputs a probability (or more accurately, the output of a sigmoid layer) with which to add the edge to the graph.
The constructor for the graph takes the parameter N upon construction, where N is the number of nodes in the graph it is meant to construct.
The network itself for this problem was fairly simple, with just a few fully connected layers, because I only considered relatively small graphs as counterexamples.
I imagine larger values of N might need larger models to accurately capture all the relevant information.

.. code-block:: python

    class ProjNet(nn.Module):
    def __init__(self, N):
        super(ProjNet, self).__init__()
        self.fc1 = nn.Linear(N*(N-1), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

Constructing Graphs from the Neural Network
-------------------------------------------

Let us call each graph construction a single "session". We begin an iteration by performing *k* sessions using the model.
*k* is actually a fairly important hyperparameter, and is the source of the largest bottle in terms of work done, in the entire model. To see why, let us think about how
big a session needs to be.

A session has N(N-1)/2 decisions, one for each edge in the graph, and we want a record of the state at each decision to train the model on, which is a vector of length N(N-1).
The total size of a session is therefore N\*N\*(N-1)\*(N-1)/2, and which is why it's slow. The original codebase did this step extremely inefficiently, with multiple redundant copies
and processing, making it a massive bottleneck. I redid the programming for this entire part, leading to large improvements in the speed of this code.

.. figure:: images/constr.png
    :width: 800

    Overview of the Graph Construction

Give the Model an Empty Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by giving the model a vector of all 0s, with only the N(N-1)/2th element set to be 1, to demarcate that we are considering the first edge currently.
We also record this state as the first part of the session.

Model outputs a probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model will take in the vector, and output a real number between 0 and 1, let's called it *p*\ , which we will treat as a probability that the edge is added or not.


Adding the edge to the graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now the graph has outputted a probability *p*, and we want to add this edge to the graph with probability *p*, which is fairly to simple to achieve with code like the following:

.. code-block:: python

    if np.random.rand() < p:
        action = 1
    else:
       action = 0

However, what would happen, and the reason the paper took so many iterations, was that *p* would end up being too extreme, and the model would get stuck generating
the same sort of graphs, essentially in a non-optimal pit.

My idea was to introduce something that slowly pushed *p* towards 0.5 over time, if the learning plateaus. I called this hyperparameter entropy in my code, but I later
learned from Professor Redmon that temperature (for Sigmoid) performs an extremely similar function. Here is what the code looked like after I applied my idea and formula:

.. code-block:: python

    if np.random.rand() < 0.5 + (p - 0.5)*entropy:
        action = 1
    else:
         action = 0

This essentially pushes *p* towards 0.5 by (1 - entropy) percentage. So if entropy was 0.9, the difference between 0.5 and *p* would decrease by 10 percent. This definitely aided
in removing pits, as we will demonstrate in my results.

Additionally, we record the action made by our code, and save it paired with the state inputted to the model.

Feed the Graph to the Model again
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we just considred the ith edge, counting from 0, we 0 out the (i + N(N-1)/2)th position, and set (i + 1 + N(N-1)/2)th position to 1, to demarcate to the model that we are now considering
the next edge. Before passing the state to the model, we record the state as the next step in the decision.

Graph is finished constructing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We repeat this cycle until all edges in the graph have been considered, and do the same for each session. Once all the sessions have been completed, we go to the next step.

Note: Technically I do all sessions simultaneously, passing a k*N(N-1) sized matrix to the model, but that's more of an implementation detail.

Scoring and Filtering Bad Graphs
--------------------------------

Once all the graphs are constructed, we first score them all using our scoring funciton *f*\ . Before any filtering, we also add in any really strong graphs from previous iterations.
After those are added, we select the top *s*-percentile of graphs, and discard the rest. We also save the top *t*-percentile of graphs for future graphs. This is also in case some extremely strong
graphs are generated.

Note that when I say I "save/select" the graphs, I'm attaching the final graph's score to every state that occured during that graph's construction, to teach the model what to do at each step in the future.

Train on the Good Graphs
------------------------
We take the selected graphs and train the Neural Network on them. It's a fairly simple and straightforward training loop.

.. code-block:: python

    def train(net, x, y, epochs=1000, lr=0.001, momentum=0.9, decay=0.0, verbose=1):
      net.to(device)
      y = torch.unsqueeze(y, 1)
      inputs, labels = x.to(device), y.to(device)
      losses = []
      criterion = nn.BCELoss()
      optimizer = optim.Adam(net.parameters(), lr=lr)
      for epoch in range(epochs):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
      return

Finding a Counterexample
------------------------

At some point during the iterations, we will find a graph that breaks the bound we were trying to achieve. Once this happens, the loop stops, and we output a visualization of the graph, along with
any relevant information.

Why this Algorithm
------------------
The issue with most reinforcement learning algorithms is they use an intermediate scoring function or heuristic to train on intermediate states. With this problem, we only get a score after all
the decisions are already made, making the best source of learning unavailable. The Deep Cross-Entropy algorithm outlined above only uses the final score to train all intermediate states
and helps teach the model what to do for those mid-session states, without having any intermediate reward function.
