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

// Laplace's 2D solver
// Using the Jacobi's iterative method

double boundary_condition(double x, double y) {
  // Example: u(x,y) = sin(π*x) * sinh(π*y) on the boundary
  // This is an analytical solution to Laplace's equation
  return sin(M_PI * x) * sinh(M_PI * y);
}

void initialize_domain(
    int nx, int ny, double u[nx][ny], double x_min, double y_min, double x_max, double y_max) {
  double dx = (x_max - x_min) / (nx - 1);
  double dy = (y_max - y_min) / (nx - 1);

  // Initialize all interior points to zero
  for(int i = 0; i < nx; i++) {
    for(int j = 0; j < ny; j++) {
      u[i][j] = 0.0;
    }
  }

  // Set boundary conditions
  for(int i = 0; i < nx; i++) {
    double x = x_min + i * dx;
    u[i][0] = boundary_condition(x, y_min);
    u[i][ny - 1] = boundary_condition(x, y_max);
  }

  for(int j = 0; j < ny; j++) {
    double y = y_min + j * dy;
    u[0][j] = boundary_condition(x_min, y);
    u[nx - 1][j] = boundary_condition(x_max, y);
  }
}

void Jacobi_method(int nx, int ny, size_t max_iter, double tolerance, double u[nx][ny]) {
  double error, max_error;
  double u_new[nx][ny];
  int iter;

  // Copy initial values to u_new
  for(int i = 0; i < nx; i++) {
    for(int j = 0; j < ny; j++) {
      u_new[i][j] = u[i][j];
    }
  }

  for(size_t iter = 0; iter < max_iter; iter++) {
    max_error = 0.0;

    // Update interior points
    for(int i = 1; i < nx - 1; i++) {
      for(int j = 1; j < ny - 1; j++) {
        // Apply finite difference stencil (discrete Laplacian)
        u_new[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]);

        error = fabs(u_new[i][j] - u[i][j]);
        if(error > max_error) {
          max_error = error;
        }
      }
    }

    // Copy new values to u
    for(int i = 1; i < nx - 1; i++) {
      for(int j = 1; j < ny - 1; j++) {
        u[i][j] = u_new[i][j];
      }
    }

    // Check for convergence
    if(max_error < tolerance) {
      printf("Solution converged after %d iterations\n", iter + 1);
      break;
    }

    // Print progress every 500 iterations
    if(iter % 500 == 0) {
      printf("Iteration %d, max error: %e\n", iter, max_error);
    }
  }

  if(iter >= max_iter) {
    printf("Solution did not converge after %zu iterations, max error: %e\n", max_iter, max_error);
  }
}

void calculate_velocity_field(int nx,
                              int ny,
                              double potential[nx][ny],
                              double u_velocity[nx][ny],
                              double v_velocity[nx][ny],
                              double x_min,
                              double y_min,
                              double x_max,
                              double y_max) {
  double dx = (x_max - x_min) / (nx - 1);
  double dy = (y_max - y_min) / (ny - 1);

  // Calculate velocities using central differences for interior points
  for(int i = 1; i < nx - 1; i++) {
    for(int j = 1; j < ny - 1; j++) {
      // u = dφ/dx using central difference
      u_velocity[i][j] = (potential[i + 1][j] - potential[i - 1][j]) / (2.0 * dx);

      // v = dφ/dy using central difference
      v_velocity[i][j] = (potential[i][j + 1] - potential[i][j - 1]) / (2.0 * dy);
    }
  }

  // Handle boundaries with forward/backward differences
  // Left and right boundaries
  for(int j = 1; j < ny - 1; j++) {
    // Left boundary (i=0) - forward difference
    u_velocity[0][j] = (potential[1][j] - potential[0][j]) / dx;

    // Right boundary (i=nx-1) - backward difference
    u_velocity[nx - 1][j] = (potential[nx - 1][j] - potential[nx - 2][j]) / dx;
  }

  // Top and bottom boundaries
  for(int i = 1; i < nx - 1; i++) {
    // Bottom boundary (j=0) - forward difference
    v_velocity[i][0] = (potential[i][1] - potential[i][0]) / dy;

    // Top boundary (j=ny-1) - backward difference
    v_velocity[i][ny - 1] = (potential[i][ny - 1] - potential[i][ny - 2]) / dy;
  }

  // Corner points - use adjacent values
  u_velocity[0][0] = u_velocity[1][0];
  v_velocity[0][0] = v_velocity[0][1];

  u_velocity[nx - 1][0] = u_velocity[nx - 2][0];
  v_velocity[nx - 1][0] = v_velocity[nx - 1][1];

  u_velocity[0][ny - 1] = u_velocity[1][ny - 1];
  v_velocity[0][ny - 1] = v_velocity[0][ny - 2];

  u_velocity[nx - 1][ny - 1] = u_velocity[nx - 2][ny - 1];
  v_velocity[nx - 1][ny - 1] = v_velocity[nx - 1][ny - 2];
}
