# Homework 2

This homework used the C's FANN library for neural networks, please install it and include it in the project's include and library link. Then run:

```bash
make all
make run {laplace | bernoulli}
```

Where choosing laplace will run the laplace's NN simulation and bernoulli will output the table shown below in section Bernoulli's ODE. The respective NNs are stored in FANN's `.net` format.

## Bernoulli's ODE

The following Bernoulli ODE was solved with the numerical RK4 implementation and then an NN:

```tex
P(x) = 4/x
Q(x) = x^3
n = 2
y(2) = -1
x > 0
```

The numerical implementation is tested in `tests.c` under `testRK()`, and the neural network get triggered in `bernoulli.c`, the following table is a showcase of differences between the NN's inference and RK4 solutions:

```cli
Inference results for dy/dx = x^3*y^2 - (4/x)*y^2:
------------------------------------------------
   x   |      NN y     |      RK4 y     
-------+---------------+----------------
 3.00  | -0.0269486606 | -0.0263820000
 3.01  | -0.0264359117 | -0.0258490000
 3.02  | -0.0259379745 | -0.0253300000
 3.03  | -0.0254544020 | -0.0248240000
 3.04  | -0.0249847472 | -0.0243320000
 3.05  | -0.0245284140 | -0.0238510000
 3.06  | -0.0240853727 | -0.0233830000
```

The numerical values were solved from x = 2 to x = 3.

## Laplace's 2D PDE

The following Laplace's 2D PDE was solved using the Jacobi numerical method:

```tex
Example Laplace: u(x,y) = sin(π*x) * sinh(π*y) on the boundary
This is an analytical solution to Laplace's equation
```

The solution was numerically calculated on the domain `x ∈ [0,1]`, `y ∈ [0,1]`, and its simulation can be seen by running the function `testJacobiSimulation()` - that is the numerical one.

To see the NN's generated scalar field (which is what the NN inferes, then the `u` and `v` velocities get numerically calculated), you can run `make run laplace`, which will infere the solution and display it in the CLI on the domain `x ∈ [0.5,1.0]`, `y ∈ [0.5,1.0]`.

> The reason for the overlapping domain, is due to poor accuracy of the NN.

Another word about the simulation, its source code can be found in `particles.c`, it simulates the particles by assigning them different characters based on its velocity:

```c
// Use different characters based on velocity magnitude
double vel_mag = sqrt(particles[p].u * particles[p].u + particles[p].v * particles[p].v);
char particle_char = '.';
if(vel_mag > 0.5) particle_char = 'o';
if(vel_mag > 1.0) particle_char = 'O';
if(vel_mag > 1.5) particle_char = '@';
```
