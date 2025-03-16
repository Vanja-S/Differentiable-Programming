#include "tests.h"

#include "particles.h"

// Example functions:
// P(x) = 4/x
// Q(x) = x^3
// n = 2
// y(2) = -1
// x > 0

double fP(double x) { return 4 / x; }

double fQ(double x) { return x * x * x; }

void testRK() {
  const double TOLERANCE = 1e-5;

  Points2D results = RK4(fP, fQ, 2, 2, -1, 3, 0.1);
  printf("Testing RK4 function...");
  printf("\tP(x) = 4/x\n\tQ(x) = x^3\n\tn = 2\n\tx_0 = -1\n\ty_0=2\n\th = 0.1\n");

  printf("Expected: \n");
  printf("| %-4s | %-5s | %-8s |\n", "step", "x", "y");
  printf("|------|-------|----------|\n");
  printf("| %-4d | %-5.2f | %-7.5f |\n", 0, 2.0, -1.00000);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 1, 2.1, -0.45513);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 2, 2.2, -0.26784);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 3, 2.3, -0.17537);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 4, 2.4, -0.12238);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 5, 2.5, -0.08917);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 6, 2.6, -0.06706);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 7, 2.7, -0.05169);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 8, 2.8, -0.04063);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 9, 2.9, -0.03247);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 10, 3.0, -0.02630);
  double x[] = {2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0};
  double y[] = {-1.00000, -0.45513, -0.26784, -0.17537, -0.12238, -0.08917,
                -0.06706, -0.05169, -0.04063, -0.03247, -0.02630};
  Points2D expected = (Points2D){.x = x, .y = y};

  printf("Computed: \n");
  printf("| %-4s | %-5s | %-8s |\n", "step", "x", "y");
  printf("|------|-------|----------|\n");
  for(size_t i = 0; i < 11; i++) {
    printf("| %-4zu | %-5.2f | %-8.5f |\n", i, results.x[i], results.y[i]);
    assert(approx_equal(expected.x[i], results.x[i], TOLERANCE));
    assert(approx_equal(expected.y[i], results.y[i], TOLERANCE));
  }
}

// Example function:
// u(x,y) = sin(π*x) * sinh(π*y)

void testJacobi() {
  const double TOLERANCE = 1e-5;

  int nx = 10;
  int ny = 10;

  // double dx = (1.0 - 0.0) / (nx - 1);
  // double dy = (1.0 - 0.0) / (ny - 1);

  double u[nx][ny];
  initialize_domain(nx, ny, u, 0.0, 0.0, 1.0, 1.0);
  Jacobi_method(nx, ny, 10, TOLERANCE, u);

  printf("Testing Jacobi function...");
  printf("\tu(x,y) = sin(π*x) * sinh(π*y)\n\tx ∈ [0,1]\n\ty ∈ [0,1]\n\titerations = 10\n");

  printf("Computed Solution:\n");
  printf("    ");
  for(size_t j = 0; j < 10; j++) {
    printf("  j=%-4zu ", j);
  }
  printf("\n");

  for(size_t i = 0; i < 10; i++) {
    printf("i=%-2zu ", i);
    for(size_t j = 0; j < 10; j++) {
      printf("%8.4f ", u[i][j]);
    }
    printf("\n");
  }
}

void testJacobiSimulation() {
  const double TOLERANCE = 1e-5;

  int nx = 10;
  int ny = 10;

  double u[nx][ny];
  double u_velocity[nx][ny], v_velocity[nx][ny];
  printf("Initializing domain...\n");
  initialize_domain(nx, ny, u, 0.0, 0.0, 1.0, 1.0);
  printf("Solving Laplace's equation by the Jacobi method...\n");
  Jacobi_method(nx, ny, 100, TOLERANCE, u);
  printf("Calculating velocity field...\n");
  calculate_velocity_field(nx, ny, u, u_velocity, v_velocity, 0.0, 0.0, 1.0, 1.0);

  printf("Initializing particles...\n");
  int num_particles = 90;
  Particle particles[num_particles];

  printf("Starting flow simulation...\n");
  initialize_particles(particles, num_particles, 0.0, 0.0);

  printf("Starting flow simulation...\n");

  size_t simulation_steps = 150;
  double time_step = 0.01;
  for(size_t step = 0; step < simulation_steps; step++) {
    update_particles(nx, ny, particles, num_particles, u_velocity, v_velocity, time_step, 0.0, 0.0,
                     1.0, 1.0);
    draw_particles(nx, ny, particles, num_particles, 0.0, 0.0, 1.0, 1.0);
  }
}
