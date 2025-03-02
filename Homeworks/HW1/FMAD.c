#include <assert.h>
#include <math.h>
#include <stdio.h>

// Types
typedef struct {
  double realComponent;
  double epsilonComponent;
} Dual;

typedef Dual (*ADFunction)(Dual *inputs, int n);

// Dual Numbers arithmetic

Dual addition(Dual x, Dual y) {
  Dual r;
  r.realComponent = x.realComponent + y.realComponent;
  r.epsilonComponent = x.epsilonComponent + y.epsilonComponent;
  return r;
}

Dual multiplication(Dual x, Dual y) {
  Dual r;
  r.realComponent = x.realComponent * y.realComponent;
  r.epsilonComponent = (x.realComponent * y.epsilonComponent) +
                       (x.epsilonComponent * y.realComponent);
  return r;
}

Dual division(Dual x, Dual y) {
  Dual r;
  r.realComponent = x.realComponent / y.realComponent;
  r.epsilonComponent = (x.epsilonComponent * y.realComponent -
                        x.realComponent * y.epsilonComponent) /
                       pow(y.realComponent, 2);
  return r;
}

Dual powDual(Dual x, double n) {
  Dual r;
  r.realComponent = pow(x.realComponent, n);
  r.epsilonComponent = n * pow(x.realComponent, n - 1) * x.epsilonComponent;
  return r;
}

Dual sinDual(Dual x) {
  Dual r;
  r.realComponent = sin(x.realComponent);
  r.epsilonComponent = cos(x.realComponent) * x.epsilonComponent;
  return r;
}

Dual cosDual(Dual x) {
  Dual r;
  r.realComponent = cos(x.realComponent);
  r.epsilonComponent = -sin(x.realComponent) * x.epsilonComponent;
  return r;
}

// Gradient and eval functions

int approx_equal(double a, double b, double tolerance) {
  return fabs(a - b) < tolerance;
}

double eval(ADFunction f, const double *x, int n) {
  Dual inputs[n];
  for(int i = 0; i < n; i++) {
    inputs[i].realComponent = x[i];
    inputs[i].epsilonComponent = 0.0;
  }

  Dual result = f(inputs, n);

  return result.realComponent;
}

/// @brief Calculates the numerical gradient of a multivariate function f with
/// the given input *x, using the Dual number system
/// @param f function pointer to an arbitrary function
/// @param *x array of numerical values
/// @param n size of the x array
/// @param *gradOut pointer to the gradient array which will be modified
/// @return
void gradient(ADFunction f, const double *x, int n, double *gradOut) {
  for(int i = 0; i < n; i++) {
    Dual inputs[n];
    for(int j = 0; j < n; j++) {
      inputs[j].realComponent = x[j];
      inputs[j].epsilonComponent = (j == i) ? 1.0 : 0.0;
    }

    Dual result = f(inputs, n);

    gradOut[i] = result.epsilonComponent;
  }
}

// Test functions

// f(x,y) = 3x + 4y + 5
Dual f1(Dual *in, int n) {
  Dual x = in[0];
  Dual y = in[1];

  Dual term1 = multiplication((Dual){3.0, 0.0}, x);
  Dual term2 = multiplication((Dual){4.0, 0.0}, y);
  Dual term3 = {5.0, 0.0};

  return addition(addition(term1, term2), term3);
}

// f(x,y) = 2sin(x) + 3cos(y)
Dual f2(Dual *in, int n) {
  Dual x = in[0];
  Dual y = in[1];

  Dual term1 = multiplication((Dual){2.0, 0.0}, sinDual(x));
  Dual term2 = multiplication((Dual){3.0, 0.0}, cosDual(y));

  return addition(term1, term2);
}

// f(x,y,z) = 5x + 3y + 4xyz
Dual f3(Dual *in, int n) {
  Dual x = in[0];
  Dual y = in[1];
  Dual z = in[2];

  Dual term1 = multiplication((Dual){5.0, 0.0}, x);
  Dual term2 = multiplication((Dual){3.0, 0.0}, y);

  Dual xyz = multiplication(x, y);
  xyz = multiplication(xyz, z);
  Dual term3 = multiplication((Dual){4.0, 0.0}, xyz);

  return addition(addition(term1, term2), term3);
}

// f(x,y) = 3xy + 5
Dual f4(Dual *in, int n) {
  Dual x = in[0];
  Dual y = in[1];

  Dual term1 = multiplication(multiplication((Dual){3.0, 0.0}, x), y);
  Dual term2 = {5.0, 0.0};

  return addition(term1, term2);
}

// f(x,y,z) = sin(sin(x)) + 2z^2 + 5cos(y)
Dual f5(Dual *in, int n) {
  Dual x = in[0];
  Dual y = in[1];
  Dual z = in[2];

  Dual term1 = sinDual(sinDual(x));
  Dual term2 = multiplication((Dual){2.0, 0.0}, powDual(z, 2));
  Dual term3 = multiplication((Dual){5.0, 0.0}, cosDual(y));

  return addition(addition(term1, term2), term3);
}

int main(int argc, char const **args) {
  printf(
      "All test cases use either a 2D point (1.0, 2.0) or a 3D point (1.0, "
      "2.0, 3.0).\n\n");

  double point3[3] = {1.0, 2.0, 3.0};
  double point2[2] = {1.0, 2.0};

  double grad2[2];
  double grad3[3];

  const double TOLERANCE = 1e-5;

  // f1
  printf("Test case 1: f(x,y) = 3x + 4y + 5\n");
  double val = eval(f1, point2, 2);
  gradient(f1, point2, 2, grad2);
  printf("\tExpected value at f(1.0, 2.0): 16\n");
  printf("\tComputed value at f(1.0, 2.0): %f\n\n", val);

  printf("\tExpected gradient at ∇f(1.0, 2.0): [3, 4]\n");
  printf("\tComputed gradient at ∇f(1.0, 2.0): [%f, %f]\n", grad2[0], grad2[1]);

  assert(val == 16.0);
  assert(grad2[0] == 3 && grad2[1] == 4);
  printf("Test Passed!\n\n");

  // f2
  printf("Test case 2: f(x,y) = 2sin(x) + 3cos(y)\n");
  val = eval(f2, point2, 2);
  gradient(f2, point2, 2, grad2);
  printf("\tExpected value at f(1.0, 2.0): 0.43450\n");
  printf("\tComputed value at f(1.0, 2.0): %.5f\n\n", val);

  printf("\tExpected gradient at ∇f(1.0, 2.0): [1.08060, -2.72789]\n");
  printf("\tComputed gradient at ∇f(1.0, 2.0): [%.5f, %.5f]\n", grad2[0],
         grad2[1]);

  assert(approx_equal(val, 0.43450, TOLERANCE));
  assert(approx_equal(grad2[0], 1.08060, TOLERANCE) &&
         approx_equal(grad2[1], -2.72789, TOLERANCE));
  printf("Test Passed!\n\n");

  // f3
  printf("Test case 3: f(x,y,z) = 5x + 3y + 4xyz\n");
  val = eval(f3, point3, 3);
  gradient(f3, point3, 3, grad3);
  printf("\tExpected value at f(1.0, 2.0, 3.0): 35\n");
  printf("\tComputed value at f(1.0, 2.0, 3.0): %.5f\n\n", val);

  printf("\tExpected gradient at ∇f(1.0, 2.0, 3.0): [29, 15, 8]\n");
  printf("\tComputed gradient at ∇f(1.0, 2.0, 3.0): [%.5f, %.5f, %.5f]\n",
         grad3[0], grad3[1], grad3[2]);

  assert(approx_equal(val, 35, TOLERANCE));
  assert(approx_equal(grad3[0], 29, TOLERANCE) &&
         approx_equal(grad3[1], 15, TOLERANCE) &&
         approx_equal(grad3[2], 8, TOLERANCE));
  printf("Test Passed!\n\n");

  // f4
  printf("Test case 4: f(x,y) = 3xy + 5\n");
  val = eval(f4, point2, 2);
  gradient(f4, point2, 2, grad2);
  printf("\tExpected value at f(1.0, 2.0): 11\n");
  printf("\tComputed value at f(1.0, 2.0): %.5f\n\n", val);

  printf("\tExpected gradient at ∇f(1.0, 2.0): [6, 3]\n");
  printf("\tComputed gradient at ∇f(1.0, 2.0): [%.5f, %.5f]\n", grad2[0],
         grad2[1]);

  assert(approx_equal(val, 11, TOLERANCE));
  assert(approx_equal(grad2[0], 6, TOLERANCE) &&
         approx_equal(grad2[1], 3, TOLERANCE));
  printf("Test Passed!\n\n");

  // f5
  printf("Test case 4: f(x,y,z) = sin(sin(x)) + 2z^2 + 5cos(y)\n");
  val = eval(f5, point3, 3);
  gradient(f5, point3, 3, grad3);
  printf("\tExpected value at f(1.0, 2.0, 3.0): 16.66488\n");
  printf("\tComputed value at f(1.0, 2.0, 3.0): %.5f\n\n", val);

  printf("\tExpected gradient at ∇f(1.0, 2.0, 3.0): [0.36003, -4.54648, 12]\n");
  printf("\tComputed gradient at ∇f(1.0, 2.0, 3.0): [%.5f, %.5f, %.5f]\n",
         grad3[0], grad3[1], grad3[2]);

  assert(approx_equal(val, 16.66488, TOLERANCE));
  assert(approx_equal(grad3[0], 0.36003, TOLERANCE) &&
         approx_equal(grad3[1], -4.54648, TOLERANCE) &&
         approx_equal(grad3[2], 12, TOLERANCE));
  printf("Test Passed!\n\n");

  return 0;
}