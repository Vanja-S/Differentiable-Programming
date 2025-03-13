#include "utils.h"

// Other utils
void append(double** array, int* size, double value) {
  *size += 1;
  *array = realloc(*array, (*size) * sizeof(double));
  if(*array == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  (*array)[*size - 1] = value;
}

int approx_equal(double a, double b, double tolerance) { return fabs(a - b) < tolerance; }

// 4th order RK method
// We rewrite Bernoulli's Equation into the form which RK4 expects:
//
// dy/dx = Q(x)y^n - P(x)y
//
// where f(x,y) = Q(x)y^n − P(x)y.
// We choose x_0 = 0 and y_0 = 0 arbitrarily.

Points2D RK4(Function P, Function Q, double n, double x0, double y0, double x_end, double h) {
  // x and y value vectors
  double* x_values = NULL;
  int size_x = 0;
  double* y_values = NULL;
  int size_y = 0;

  double x = x0;
  double y = y0;

  // Add initial values
  append(&x_values, &size_x, x0);
  append(&y_values, &size_y, y0);

  while(x < x_end) {
    if(x + h > x_end) {
      h = x_end - x;
    }

    double k1 = Q(x) * pow(y, n) - P(x) * y;
    double k2 = Q(x + h / 2) * pow(y + h / 2 * k1, n) - P(x + h / 2) * (y + h / 2 * k1);
    double k3 = Q(x + h / 2) * pow(y + h / 2 * k2, n) - P(x + h / 2) * (y + h / 2 * k2);
    double k4 = Q(x + h) * pow(y + h * k3, n) - P(x + h) * (y + h * k3);

    y += (h / 6.0f) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    x += h;

    append(&x_values, &size_x, x);
    append(&y_values, &size_y, y);
  }

  return (Points2D){.x = x_values, .y = y_values};
}
