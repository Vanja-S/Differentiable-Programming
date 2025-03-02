# Homework 1

All expected values and derivatives were calculated using WolframAlpha's calculator.

## Numerical Differentiation

The function `numericalDifferentiation` takes in 4 parameters:

- A function pointer to a vector-valued function
- A vector of variables at which to evaluate the function's derivate
- The size of the variable vector
- Step h is in the derivative's definition.

It uses the centralized derivative definition for more precise calculations. It calculates the gradient of every function passed in and compiles the Jacobian matrix. Most of the test cases have only one row in the Jacobian, i.e., a gradient.

## Forward Mode Automatic Differentiation (FMAD)

The implemented structure:

```c
typedef struct {
  double real component;
  double epsilonComponent;
} Dual;
```

Represents the Dual number system. I have implemented basic functions to represent addition, multiplication, division, exponentiation, sin and cos functions in the Dual system.

A few hardcoded functions are used to calculate their gradient and value at some point.
