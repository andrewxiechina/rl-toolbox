# Deep Q-network (DQN)

## Background

Deepmind published a paper on a Deep Neural Networks for Reinforcement Learning. The system was able to master a diverse range of Atari 2600 games to superhuman level with only the raw pixels and score as inputs.

This work represents the first demonstration of a general-purpose agent that is able to continually adapt its behavior without any human intervention. 

## Algorithm
```text
Initialize replay memory D (memory capacity is set to N)
Initialize action-value function Q with random weights 

for episode = 1 -> M do 
	Initialise state sequence s1 = {x1} 
	Preprocess states into φ1 = φ(s1) 
	for t = 1 -> T do
		With probability ε select a random action a 
		otherwise select at = max(a) Q∗(φ(st), a; θ)

		Execute at in emulator and observe reward rt and image xt+1 
		Set st+1 = st, at, xt+1
		Preprocess φt+1 = φ(st+1)
		Store transition (φt, at, rt, φt+1) in D

		Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
		
		yj = rj 	for terminal φj+1
		yj = rj + γ max(a)′ Q(φj+1, a′; θ) 	for non-terminal φj+1
		
		Perform a gradient descent step on (yj − Q(φj , aj; θ))^2
	end for
end for
```

## Citation
- **Deepmind**	[https://deepmind.com/research/dqn/](https://deepmind.com/research/dqn/)
- **Paper** 	[https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
