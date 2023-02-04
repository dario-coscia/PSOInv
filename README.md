# PSOInv
Particle Swarm Optimisation for Inverse Problems


## Table of contents
* [Description](#description)
* [Dependencies and installation](#dependencies-and-installation)
	* [Installing from source](#installing-from-source)
* [Mathematical Summary](#mathematical-summary)
    * [Particle Swarm Optimization](#particle-swarm-optimization)
    * [PSO Implementation](#pso-implementation)
    * [Inverse Problems](#inverse-problems)
* [Examples and Tutorials](#examples-and-tutorials)
* [Tests](#tests)
* [Authors and contributors](#authors-and-contributors)
* [How to contribute](#how-to-contribute)
	* [Submitting a patch](#submitting-a-patch)
* [License](#license)
* [References](#references)

## Description
**PSOinv** is a Python package, built on Python, providing an easy interface to deal with inverse problems optimized by particle swarm optimization. Based on Numpy, PSOinv offers a simple and intuitive way to formalize a specific inverse problem and solve it using particle swarm optimization.

## Dependencies and installation
**pso** requires requires `numpy`, `sphinx` (for the documentation) and `pytest` (for local test). The code is tested for Python 3, while compatibility of Python 2 is not guaranteed anymore. It can be installed directly from the source code.

### Installing from source
The official distribution is on GitHub, and you can clone the repository using
```bash
> git clone https://github.com/dario-coscia/PSOInv
```
You can also install it using pip via
```bash
> python -m pip install git+https://github.com/dario-coscia/PSOInv
```

## Mathematical Summary
In computational science, optimization is the process of finding the points that minimize a function, or set of functions, over a set. In particular, particle swarm optimization (PSO) is a particular type of optimization method. PSO finds a optimization problem solution by iteratively trying to improve a candidate solution with regard to a given measure of quality (fitness). In this repository we share the code for applying PSO to solve inverse problems.

### Particle Swarm Optimization
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
from psoinv import PSO
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

### Inverse Problems
An inverse problem is a general framework that is used to convert observed measurements into information about a physical object or system that one is interested in. 

In our case we consider a dynamical system described by a set of parametrical differential equation (DE), where we have measurements of the physical solution and we aim to find the parameters of the governing differential equation. Formally, consider the general form of a differential equation (for system of differential equation the formulation is analogous):

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

## Examples and Tutorials
The directory `problems` contains some examples showing how to use solve inverse problems using **psoinv**.

## Tests

In order to run the tests on the package `pytest` is needed.

To tests the implementation, run on the main directory the command:

```
pytest
```

## Authors and contributors
**psoinv** is currently developed and mantained by [Data Science and Scientific Computing](https://dssc.units.it/) master students:
* [Dario Coscia](https://github.com/dario-coscia)

Contact us by email for further information or questions about **veni**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!


## How to contribute
We'd love to accept your patches and contributions to this project. There are just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use 4
     spaces to indent the code. The easy way is to run on your bash the provided
     script: ./code_formatter.sh. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.
     
  4. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  5. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request

## Citations
If you are considering using **psoinv** on your reaserch please cite us:

```bash
Coscia, D. (2023). psoinv (Version 0.0.1) [Computer software]. https://github.com/dario-coscia/PSOInv
```
You can also download the bibtex format from the citation widget on the sidebar

## License

See the [LICENSE](LICENSE) file for license rights and limitations (MIT).

## References
To implement the package we follow these works:

* Sengupta S, Basak S, Peters RA II. Particle Swarm Optimization: A Survey of Historical and Recent Developments with Hybridization Perspectives. Machine Learning and Knowledge Extraction. 2019; 1(1):157-191. https://doi.org/10.3390/make1010010
