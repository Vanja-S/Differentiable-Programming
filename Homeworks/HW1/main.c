// Author: Vanja Stojanović
// Numerical Differentiation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Types
typedef double* (*Function)(const double *);

typedef struct VectorFunction {
  Function** inputFunctions;
  size_t n;
  size_t m;
} VectorFunction;

int approx_equal(double a, double b, double tolerance) {
    return fabs(a - b) < tolerance;
}

double* gradient(Function f, const double* vec, int n, double h) {
    double* gradient = malloc(n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        // Prepare temp vectors with derivative h subtracted/added
        double* tempVec1 = malloc(n * sizeof(double));
        memcpy(tempVec1, vec, n * sizeof(double));
        tempVec1[i] = tempVec1[i] + h;
        
        double* tempVec2 = malloc(n * sizeof(double));
        memcpy(tempVec2, vec, n * sizeof(double));
        tempVec2[i] = tempVec2[i] - h;
        
        // Centralized derivative defnition
        double* result1 = f(tempVec1);
        double* result2 = f(tempVec2);
        
        gradient[i] = (*result1 - *result2)/(2 * h);
        
        free(tempVec1);
        free(tempVec2);
        free(result1);
        free(result2);
    }
    
    return gradient;
}


/// @brief Calculates the numerical Jacobian of a vector valued function f with the given input *vec.
/// @param f VectorFunction struct containing functions 
/// @param vec array of numerical values
/// @param n size of the vec array
/// @param h Default is 0.0001 if not provided
/// @return 
double** numericalDerivative(VectorFunction* f, const double* vec, int n, double h) {
    double** jacobian = malloc(f->n * f->m * sizeof(double*));
    for (int i = 0; i < f->n * f->m; i++) {
        jacobian[i] = malloc(n * sizeof(double));
    }

    int row = 0;
    for (size_t i = 0; i < f->n; i++) {
        for (size_t j = 0; j < f->m; j++) {
            double* grad = gradient(f->inputFunctions[i][j], vec, n, h);
            memcpy(jacobian[row], grad, n * sizeof(double));
            free(grad);
            row++;
        }
    }
    
    return jacobian;
}

/// @brief Calculates the numerical Jacobian of a vector valued function f with the given input *vec. Default h is 0.0001.
/// @param *f VectorFunction struct containing functions 
/// @param *vec array of numerical values
/// @param n size of the vec array
/// @return 
double** numericalDerivativeDefaultH(VectorFunction* f, const double* vec, int n) {
    return numericalDerivative(f, vec, n, 0.0001);
}

/// @brief Free Jacobian matrix
void freeJacobian(double** jacobian, int rows) {
    for (int i = 0; i < rows; i++) {
        free(jacobian[i]);
    }
    free(jacobian);
}

/// @brief Free VectorFunction
void freeVectorFunction(VectorFunction* vf) {
    for (size_t i = 0; i < vf->n; i++) {
        free(vf->inputFunctions[i]);
    }
    free(vf->inputFunctions);
    free(vf);
}

/// @brief Setup VectorFunction with a single function
VectorFunction* setupSingleFunction(Function f) {
    VectorFunction* vf = malloc(sizeof(VectorFunction));
    vf->n = 1;
    vf->m = 1;
    vf->inputFunctions = malloc(vf->n * sizeof(Function*));
    vf->inputFunctions[0] = malloc(vf->m * sizeof(Function));
    vf->inputFunctions[0][0] = f;
    return vf;
}

// Test functions
// 1. f(x,y) = sin(x)/cos(y)
double* testFunction1(const double *input) {
    double* result = malloc(sizeof(double));
    *result = sin(input[0])/cos(input[1]);
    return result;
}

// Analytical derivatives for testFunction1:
// df/dx = cos(x)/cos(y)
// df/dy = sin(x)*sin(y)/cos(y)^2

// 2. f(x,y) = x^2 + 2*y^2
double* testFunction2(const double *input) {
    double* result = malloc(sizeof(double));
    *result = input[0] * input[0] + 2 * input[1] * input[1];
    return result;
}

// Analytical derivatives for testFunction2:
// df/dx = 2*x
// df/dy = 4*y

// 3. f(x,y) = e^(x*y)
double* testFunction3(const double *input) {
    double* result = malloc(sizeof(double));
    *result = exp(input[0] * input[1]);
    return result;
}

// Analytical derivatives for testFunction3:
// df/dx = y*e^(x*y)
// df/dy = x*e^(x*y)

// 4. f(x,y,z) = x*y + y*z + x*z
double* testFunction4(const double *input) {
    double* result = malloc(sizeof(double));
    *result = input[0] * input[1] + input[1] * input[2] + input[0] * input[2];
    return result;
}

// Analytical derivatives for testFunction4:
// df/dx = y + z
// df/dy = x + z
// df/dz = y + x

// 5. f(x,y) = ln(x^2 + y^2)
double* testFunction5(const double *input) {
    double* result = malloc(sizeof(double));
    *result = log(input[0] * input[0] + input[1] * input[1]);
    return result;
}

// Analytical derivatives for testFunction5:
// df/dx = 2*x/(x^2 + y^2)
// df/dy = 2*y/(x^2 + y^2)

int main(int argc, char const **args) {
    const double TOLERANCE = 1e-5;
    
    printf("Running test cases for numerical differentiation...\n\n");
    
    // Test Case 1: f(x,y) = sin(x)/cos(y)
    printf("Test Case 1: f(x,y) = sin(x)/cos(y)\n");
    double testVec1[2] = {1.0, 0.5};
    VectorFunction* vf1 = setupSingleFunction(testFunction1);
    
    double** jacobian1 = numericalDerivativeDefaultH(vf1, testVec1, 2);
    
    // Analytical derivatives at (1.0, 0.5)
    double df1_dx_expected = cos(1.0)/cos(0.5);
    double df1_dy_expected = sin(1.0)*sin(0.5)/pow(cos(0.5), 2);
    
    printf("  Expected: df/dx = %f, df/dy = %f\n", df1_dx_expected, df1_dy_expected);
    printf("  Computed: df/dx = %f, df/dy = %f\n", jacobian1[0][0], jacobian1[0][1]);
    
    assert(approx_equal(jacobian1[0][0], df1_dx_expected, TOLERANCE));
    assert(approx_equal(jacobian1[0][1], df1_dy_expected, TOLERANCE));
    printf("  Test passed!\n\n");
    
    // Test Case 2: f(x,y) = x^2 + 2*y^2
    printf("Test Case 2: f(x,y) = x^2 + 2*y^2\n");
    double testVec2[2] = {2.0, 3.0};
    VectorFunction* vf2 = setupSingleFunction(testFunction2);
    
    double** jacobian2 = numericalDerivativeDefaultH(vf2, testVec2, 2);
    
    // Analytical derivatives at (2.0, 3.0)
    double df2_dx_expected = 2 * 2.0;
    double df2_dy_expected = 4 * 3.0;
    
    printf("  Expected: df/dx = %f, df/dy = %f\n", df2_dx_expected, df2_dy_expected);
    printf("  Computed: df/dx = %f, df/dy = %f\n", jacobian2[0][0], jacobian2[0][1]);
    
    assert(approx_equal(jacobian2[0][0], df2_dx_expected, TOLERANCE));
    assert(approx_equal(jacobian2[0][1], df2_dy_expected, TOLERANCE));
    printf("  Test passed!\n\n");
    
    // Test Case 3: f(x,y) = e^(x*y)
    printf("Test Case 3: f(x,y) = e^(x*y)\n");
    double testVec3[2] = {0.5, 1.5};
    VectorFunction* vf3 = setupSingleFunction(testFunction3);
    
    double** jacobian3 = numericalDerivativeDefaultH(vf3, testVec3, 2);
    
    // Analytical derivatives at (0.5, 1.5)
    double df3_dx_expected = 1.5 * exp(0.5 * 1.5);
    double df3_dy_expected = 0.5 * exp(0.5 * 1.5);
    
    printf("  Expected: df/dx = %f, df/dy = %f\n", df3_dx_expected, df3_dy_expected);
    printf("  Computed: df/dx = %f, df/dy = %f\n", jacobian3[0][0], jacobian3[0][1]);
    
    assert(approx_equal(jacobian3[0][0], df3_dx_expected, TOLERANCE));
    assert(approx_equal(jacobian3[0][1], df3_dy_expected, TOLERANCE));
    printf("  Test passed!\n\n");
    
    // Test Case 4: f(x,y,z) = x*y + y*z + x*z
    printf("Test Case 4: f(x,y,z) = x*y + y*z + x*z\n");
    double testVec4[3] = {1.0, 2.0, 3.0};
    VectorFunction* vf4 = setupSingleFunction(testFunction4);
    
    double** jacobian4 = numericalDerivativeDefaultH(vf4, testVec4, 3);
    
    // Analytical derivatives at (1.0, 2.0, 3.0)
    double df4_dx_expected = 2.0 + 3.0;
    double df4_dy_expected = 1.0 + 3.0;
    double df4_dz_expected = 2.0 + 1.0;
    
    printf("  Expected: df/dx = %f, df/dy = %f, df/dz = %f\n", 
           df4_dx_expected, df4_dy_expected, df4_dz_expected);
    printf("  Computed: df/dx = %f, df/dy = %f, df/dz = %f\n", 
           jacobian4[0][0], jacobian4[0][1], jacobian4[0][2]);
    
    assert(approx_equal(jacobian4[0][0], df4_dx_expected, TOLERANCE));
    assert(approx_equal(jacobian4[0][1], df4_dy_expected, TOLERANCE));
    assert(approx_equal(jacobian4[0][2], df4_dz_expected, TOLERANCE));
    printf("  Test passed!\n\n");
    
    // Test Case 5: f(x,y) = ln(x^2 + y^2)
    printf("Test Case 5: f(x,y) = ln(x^2 + y^2)\n");
    double testVec5[2] = {2.0, 1.0};
    VectorFunction* vf5 = setupSingleFunction(testFunction5);
    
    double** jacobian5 = numericalDerivativeDefaultH(vf5, testVec5, 2);
    
    // Analytical derivatives at (2.0, 1.0)
    double df5_dx_expected = 2 * 2.0 / (2.0 * 2.0 + 1.0 * 1.0);
    double df5_dy_expected = 2 * 1.0 / (2.0 * 2.0 + 1.0 * 1.0);
    
    printf("  Expected: df/dx = %f, df/dy = %f\n", df5_dx_expected, df5_dy_expected);
    printf("  Computed: df/dx = %f, df/dy = %f\n", jacobian5[0][0], jacobian5[0][1]);
    
    assert(approx_equal(jacobian5[0][0], df5_dx_expected, TOLERANCE));
    assert(approx_equal(jacobian5[0][1], df5_dy_expected, TOLERANCE));
    printf("  Test passed!\n\n");
    
    printf("All tests passed successfully!\n");
    
    // Free allocated memory
    freeJacobian(jacobian1, vf1->n * vf1->m);
    freeJacobian(jacobian2, vf2->n * vf2->m);
    freeJacobian(jacobian3, vf3->n * vf3->m);
    freeJacobian(jacobian4, vf4->n * vf4->m);
    freeJacobian(jacobian5, vf5->n * vf5->m);
    
    freeVectorFunction(vf1);
    freeVectorFunction(vf2);
    freeVectorFunction(vf3);
    freeVectorFunction(vf4);
    freeVectorFunction(vf5);
    
    return 0;
}