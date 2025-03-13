#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Simple f: R -> R
typedef double (*Function)(double);

// 2D points structure
typedef struct {
  double* x;
  double* y;
} Points2D;

// Other utils
void append(double** array, int* size, double value);
int approx_equal(double a, double b, double tolerance);

// RK method
Points2D RK4(Function P, Function Q, double n, double x0, double y0, double x_end, double h);