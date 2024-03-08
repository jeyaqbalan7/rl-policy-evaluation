# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
The Bandit Slippery Walk problem is a Reinforcement Learning (RL) problem in which the agent must learn to navigate a slippery environment to reach the goal state.

1. we are tasked with creating an RL agent to solve the "Bandit Slippery Walk" problem.

2. The environment consists of Seven states representing discrete positions the agent can occupy.

3. The agent must learn to navigate this environment while dealing with the challenge of slippery terrain.

4. Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## STATE
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.Five Transition states / Non-terminal States including S: The starting state.

## Actions
The agent can take two actions: R (move right) and L (move left). 

The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.

## REWARD
The agent receives a reward of +1 for reaching the goal state and a reward of 0 for all other states.

## GRAPHICAL REPRESENTATION
![Graph](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/81942bf3-5a4d-455c-be48-b7aa74de98b3)

## FORMULA
![form](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/5e063a00-1ed5-4f4a-ab35-d9e9a13e1cc8)

## POLICY EVALUATION FUNCTION
~~~python

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
# code  to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma+prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
      return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing the two policies
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
~~~
## OUTPUT:
### POLICY 1
![P1](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/32634a57-1a5b-44bd-bb66-b12b8bce611e)
![P2](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/96c8e7be-40ab-4611-a9b8-3ac716ec3914)
![P3](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/3089dee7-6bba-48a4-a733-3b3180aefc2d)

### POLICY 2
![PP1](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/f2bb7a4c-0728-4b73-b7ef-236aaaf0b84f)
![PP2](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/61d21b8d-2da5-482e-b46a-3603bc87279c)
![PP3](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/d93717d1-f154-4561-bbbb-259a9bcd631c)

### COMPARISON
![C1](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/62f43928-10c2-4e81-8a93-57a06662a299)

### CONCLUSION
![CC1](https://github.com/Pravinrajj/rl-policy-evaluation/assets/117917674/be819cad-905e-4ecd-b80e-366070380f8b)

## RESULT:
Thus, This program will evaluate the given policy in the Bandit Slippery Walk environment and predict the expected reward of the policy.
