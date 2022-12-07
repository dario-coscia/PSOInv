# PSOInv
Particle Swarm Optimisation for Inverse Problems

## Summary
In computational science, optimization is the process of finding the points that minimize a function, or set of functions, over a set. In particular, particle swarm optimization (PSO) is a particular type of optimization method. PSO finds a optimization problem solution by iteratively trying to improve a candidate solution with regard to a given measure of quality (fitness). In this repository we share the code for applying PSO to solve inverse problems.

## Particle Swarm Optimization
Particle Swarm Optimization is a population-based search algorithm for solving optimization problems. The PSO algorithm maintains a swarm of **particles**, where each particle represent a possible solution. In evolutionary computation paradigms, the swarm represents the population, while the single particle and individual. 

Individuals in a particle swarm follow a very simple behavior: to emulate the success of neighboring individuals and their own successes. The collective behavior that emerges
from this simple behavior is that of discovering optimal regions of a high dimensional
search space [*Engelbrecht, Andries P. Computational intelligence: an introduction. John Wiley & Sons, 2007*].

Mathematically, given a set $\Omega\subset\mathbb{R}^d$ and a fitness function $f : \Omega \rightarrow \mathbb{R}$ to optimize, each particle $i$ is identified by its position $\pmb{x}_i(t)$ at a specific discrete time step $t$. The position of the particle is changed by the following update rule:

$$\pmb{x}_i(t+1) = \pmb{x}_i(t) + \pmb{v}_i(t),$$

where $\pmb{v}_i(t)$ is the velocity of the particle $i$ and time $t$. Initially partile are initialized uniformly with random positions in $\Omega$. The velocity update rule is composed by three factors: cognitive factor, social factor and inertia factor.

1. **Cognitive factor**: Reflects the attraction towards the best local position $\pmb{b}_i$ (for particle) found so far by the current particle (experiential knowledge of the particle).
2. **Social factor**: Reflects the attraction towards the best global position $\pmb{g}$ (between particle) found so far in the entire swarm (socially exchanged information from the
particle’s neighborhood).
3. **Inertia factor**: Reflects the ability of a particle to avoid rapid change in velocity.

In practice the update rule is written as:

$$
\pmb{v_i}(t+1) = w \cdot \pmb{v_i}(t) + c_{\rm{soc}} \cdot \pmb{r_1} \otimes (\pmb{g} - \pmb{x_i}(t)) + c_{\rm{cog}} \cdot \pmb{r_2} \otimes (\pmb{b_i} - \pmb{x_i}(t)),
$$

where:

* $\pmb{r}_1$ and $\pmb{r}_2$ are random vectors from $[0, 1]^d$, and $\otimes$ denotes the Hadamard product,
* $c_{\rm{cog}} \in \mathbb{R}^+$ is the the cognitive factor coefficient,
* $c_{\rm{soc}} \in \mathbb{R}^+$ is the the social factor coefficient,
* $w \in \mathbb{R}^+$ is the the inertia factor coefficient.

### PSO Implementation
The PSO implementation is written in `numpy`, which is the only requirement for using the `PSO` class. A simple example of how to use it is reported below:

```python
from pso import PSO
import numpy as np

# define fitness function to minimize
def sphere(x):
    return x[:, 0]**2 + x[:, 1]**2

# define bounds space + bounds velocity
bounds_space = [np.array([-10, 10], np.array([-10, 10]]
bounds_vel = [np.array([0.0001, 0.5]), np.array([0.0001, 0.5])]

# define the PSO optimizer
pso = PSO(swarm_size = 100,
          boundaries = bounds_space,
          velocities = bounds_vel,
          fit = sphere,
          velocity_rule=None,
          n_iter=1000)

# perform optimization
pso.fit()
```

By setting `velocity_rule=None` the default values for cognitive, social and inertia factor are used. Alternatively different values can be passed in form of dictionary. More information can be found on the documentation. The fit function is a simple numpy function applied to an array of size $N\times d$, with $d$ the dimension of the problem and $N$ the number of points in the domain.

## Inverse Problems
An inverse problem is a general framework that is used to convert observed measurements into information about a physical object or system that one is interested in. 

In our case we consider a dinamical system described by a set of parametrical differential equation (DE), where we have measurements of the physical solution and we aim to find the parameters of the governing differential equation. Formally, consider the general form of a differential equation (for system of differential equation the formulation is analogous):

$$
\begin{equation}
\begin{split}
    \mathcal{A}(\pmb{u}(\pmb{z});\alpha)&=\pmb{f}(\pmb{z}) \quad \pmb{z} \in \Omega\\       
    \mathcal{B}(\pmb{u}(\pmb{z}))&=\pmb{g}(\pmb{z}) \quad \pmb{z} \in \partial\Omega,   
\end{split}
\end{equation}
$$

where $\mathcal{A}$ is the mathematical differential operator, $\pmb{u}$ the solution, $\pmb{z}$ the spatio-temporal coordinates, $\pmb{f}$ the forcing term and $\alpha$ the set of paramenters. Furthemore, $\mathcal{B}$ identifies the operator indicating arbitrary initial or boundary conditions and $\pmb{g}$ the boundary function.

Let $\alpha^{real}$ the real physical parameter that, inserted in *(1)* and solved the DE, generate our data $\pmb{u}^{real}$. For a specific $\hat{\alpha}$, by solving the DE *(1)* we can find a solution $\pmb{\hat{u}}$. Hence, we can define a fitness as:

$$
f(\hat{\alpha}) = || \pmb{u}^{real} - \pmb{\hat{u}} ||^2.
$$

This function is clearly minimized when $\hat{\alpha} = \alpha^{real}$ since the solution is unique. In our research we identified each parameter with a "spatial dimension", and each particle is a vector of the DE parameters. Hence we evolved the parameters and we search for the optimial one, minimizing $f$. In the `problems` directory you can find different applications: accellerated motion, double pendulum, lodka volterra, lorentz attractor. 
