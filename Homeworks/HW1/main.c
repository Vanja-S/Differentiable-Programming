// Author: Vanja Stojanović
// Numerical Differentiation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Types
typedef double* (*Function)(const double *);

typedef struct VectorFunction {
  Function** inputFunctions;
  size_t n;
  size_t m;
} VectorFunction;

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

// 2 inputs, x and y
double* testFunction(const double *input) {
    double* result = malloc(sizeof(double));
    *result = 2.0 * M_PI + sin(input[0])/cos(input[1]);    
    return result;
}

int main(int argc, char const **args) {
    double testVec[2] = {1.0, 0.5};

    double h = 0.0001;
    VectorFunction* vf = malloc(sizeof(VectorFunction));
    vf->n = 3;
    vf->m = 2;

    vf->inputFunctions = malloc(vf->n * sizeof(Function*));
    for (size_t i = 0; i < vf->n; i++) {
        vf->inputFunctions[i] = malloc(vf->m * sizeof(Function));
    }
    for (size_t i = 0; i < vf->n; i++) {
        for (size_t j = 0; j < vf->m; j++){
            vf->inputFunctions[i][j] = testFunction;
        }
    }
    
    double** jacobian = numericalDerivativeDefaultH(vf, testVec, 2);

    printf("Jacobian:\n");
    for(size_t i = 0; i < vf->n * vf->m; i++) {
        printf("%f  %f\n", jacobian[i][0], jacobian[i][1]);
    }

    for (size_t i = 0; i < vf->n; i++) {
        free(vf->inputFunctions[i]);
    }
    free(vf->inputFunctions);
    free(vf);
    return 0;
}