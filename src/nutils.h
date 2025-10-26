#ifndef NUTILS_H
#define NUTILS_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// definition of euler's number
#define EULER_NUMBER 2.71828

void init_utils(); // initialize utilities
double sigmoid(double n); // calculate sigmoid
void *sigmoid_vector(gsl_vector *v, gsl_vector *r); // calculate sigmoid of vector (outputs to r)
gsl_vector *concatenate_vector(gsl_vector *a, gsl_vector *b); // combine (concatenate) two vectors

// utilities for printing 
void print_vector(gsl_vector *v, char *s); 
void print_matrix(gsl_matrix *m, char *s);

// utilities for generating random objects
double random_double(double range1, double range2);
void randomize_vector(gsl_vector *x, double range1, double range2);
void randomize_matrix(gsl_matrix *x, double range1, double range2);
gsl_vector *create_rand_vector(int size, double range1, double range2);
gsl_matrix *create_rand_matrix(int size1, int size2, double range1, double range2);

#endif
