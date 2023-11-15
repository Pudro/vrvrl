# VRVRL: Vehicle Routing via Reinforcement Learning

[Spreadsheet with resources](https://docs.google.com/spreadsheets/d/1Q4Wk0HqszunIwziRhL87jSH6zn_71YOMRs9Lhxso2rk/edit?usp=sharing)


## Notes
This section might be temporary; Maybe it will help us understand the code.

1. Learn to Improve (L2I) framework
- Start with a feasible solution
- Perform improvements or perturbations
- After *T* steps the algorithm stops and the solution with the minimum travelling cost is chosen
- Given a history of most recent solutions, a threshold-based rule decides whether to continue improving the current solution or to perturb it and restart with the perturbed solution

2. Improvement Controller and Operators
- Initial solution is either constructed randomly or produced by a perturbation operator
- The controller tries to improve the solution without violating any constraints. 
- With current state as input, the neural network produces a vector of action probabilities, and the weights are trained with policy gradient.

3. States
- Each state includes features from the problem instance, the solution and the running history
- The running history includes the actions that are recently taken as well as their effects

4. Actions
- Actions can be either intra-route operators or inter-route operators
- The same operator with different parameters are considered as different actions

5. Policy Network
- REINFORCE algorithm is used to update the gradient
- Given a state, a policy network outputs a list of action probabilities, one for each action
- Problem- and solution-specific input features are transformed into an embedding of length D (D = 64)
- The embedding is fed into an attention network (8 heads and 64 output units)
- The output of the attention network is concatenated with a sequence of recent actions and their effects
- The concatenated values are fed into a network of two fully connected layers (firs is 64 units and ReLU activation function; second layer uses Softmax)

6. Rewards
- There are two reward functions
- RF1 focuses on the immediate impact of the improvement operators. The rewards is +1 if the operator improves the current solution, -1 otherwise
- RF2 is advantage-based. The total distance achieved for the problem instance during the first improvement iteration is taken as a baseline. For each subsequent iterations, all operators applied during this iteration received a reward equal to the difference between the distance achieved during the iteration and the baseline
- The discount factor $\gamma$ = 1, to equally reward operators used in the same improvement iteration
