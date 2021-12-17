Project Idea
==================


Abstract
-----------

In lecture, we discussed how reinforcement learning has been applied to games like Chess and Go, and provided amazing results.
The performance of computers in these games, made me wonder where else reinforcement learning could be applied.
I discovered a paper [Wagner]_ that applied reinforcement learning to the idea of constructing graphs which disprove conjectures in combinatorics.
This can be done by thinking of the construction of a graph as a game, where each decision is deciding if a certain edge should be added to the graph
or not.
In this project, my goal was to replicate and improve on the results in the paper by reprogramming the algorithm, while adding my own modifications.

Problem Statement
------------------
Given a graph conjecture of the form: "**f**\ (G) > 0 for all graphs G", where **f** is an arbitrary scoring function, the goal is to try and
construct a graph such that **f**\ (G) <= 0. This would break the bound placed by the conjecture.

Note: The direction of the inequality and the value of the bound can easily be changed, and I picked arbitrary ones for the sake of explanation here.

Related Work
------------
All of the work in the paper is based on the work in [Wagner]_. Since I had very little prior experience with Reinforcement Learning, I focused on
trying to improve the algorithm and ideas used by Wagner. Wagner did have a code repository written in TensorFlow available for the paper, but since
my goal was to try and improve on the work, I wrote the entire algorithm ground up in PyTorch, applying improvements throughout.

.. [Wagner] Wagner, A. Constructions in Combinatorics via Neural Networks, 2021; https://arxiv.org/abs/2104.14516
