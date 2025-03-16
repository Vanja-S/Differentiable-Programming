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

// Jacobi method

void initialize_domain(
    int nx, int ny, double u[nx][ny], double x_min, double y_min, double x_max, double y_max);
void Jacobi_method(int nx, int ny, size_t max_iter, double tolerance, double u[nx][ny]);
void calculate_velocity_field(int nx,
                              int ny,
                              double potential[nx][ny],
                              double u_velocity[nx][ny],
                              double v_velocity[nx][ny],
                              double x_min,
                              double y_min,
                              double x_max,
                              double y_max);