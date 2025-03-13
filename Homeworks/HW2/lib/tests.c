#include "tests.h"

// Example functions:
// P(x) = x
// Q(x) = x^2
// n = 2
double fP(double x) { return x; }

double fQ(double x) { return x * x; }

void testRK() {
  const double TOLERANCE = 1e-5;

  Points2D results = RK4(fP, fQ, 2.0f, 0, 2, 1, 0.25f);
  printf("Testing RK4 function...");
  printf("\tP(x) = 4/sin(x)\n\tQ(x) = x^2\n\tn = 2\n\tx_0 = 0\n\ty_0=2\n\th = 0.25\n");

  printf("Expected: \n");
  printf("| %-4s | %-5s | %-7s |\n", "step", "x", "y");
  printf("|------|-------|---------|\n");
  printf("| %-4d | %-5.2f | %-7.5f |\n", 0, 0.00, 2.00000);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 1, 0.25, 1.95846);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 2, 0.50, 1.91292);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 3, 0.75, 1.98175);
  printf("| %-4d | %-5.2f | %-7.5f |\n", 4, 1.00, 2.41706);
  double x[] = {0.00, 0.25, 0.50, 0.75, 1.00};
  double y[] = {2.00000, 1.95846, 1.91292, 1.98175, 2.41706};
  Points2D expected = (Points2D){.x = x, .y = y};

  printf("Computed: \n");
  printf("| %-4s | %-5s | %-7s |\n", "step", "x", "y");
  printf("|------|-------|---------|\n");
  for(size_t i = 0; i < 5; i++) {
    printf("| %-4zu | %-5.2f | %-7.5f |\n", i, results.x[i], results.y[i]);
    assert(approx_equal(expected.x[i], results.x[i], TOLERANCE));
    assert(approx_equal(expected.y[i], results.y[i], TOLERANCE));
  }
}