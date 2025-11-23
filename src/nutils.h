#ifndef NUTILS_H
#define NUTILS_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// definition of euler's number
#define EULER_NUMBER 2.71828

void init_utils(); // initialize utilities

// math utilities
double sigmoid(double n); // calculate sigmoid
void sigmoid_vector(gsl_vector *v, gsl_vector *r); // calculate sigmoid of vector (outputs to r)
void tanh_vector(gsl_vector *v, gsl_vector *r); // calculate tanh of vector
void concatenate_vector(gsl_vector *a, gsl_vector *b, gsl_vector *r); // combine (concatenate) two vectors
void hdm_vector(gsl_vector *a, gsl_vector *b, gsl_vector *r); // get hadamard product of two vectors a and b and output to r
double mse(double a, double b); // mean squared error on 2 values
double mse_vector(gsl_vector *a, gsl_vector *b); // perform mean squared error on multiple values
void mul_vector(gsl_vector *a, double c, gsl_vector *r); // multiply elements of vector a with constant c, c*a = r.

// utilities for series of vectors
gsl_vector **series_vectors(int size, int n, double range1i, double range2i, double range1v, double range2v); // create an array of n vectors of length size for simulating graphs. 
// range1i, range2i = minimum, maximum of randomly generated initial values of vector.
// range1v, range2v = minimum, maximum of how much the randomly generated subsequent vectors will change. 
void free_series_vectors(gsl_vector **v, int n); // frees series vectors
void print_series_vectors(gsl_vector **v, int n, char *s); // print series of n vectors, s = title string

// utilities for printing, s = title string
void print_vector(gsl_vector *v, char *s); 
void print_matrix(gsl_matrix *m, char *s);

// utilities for generating random objects
double random_double(double range1, double range2);
void randomize_vector(gsl_vector *x, double range1, double range2);
void randomize_matrix(gsl_matrix *x, double range1, double range2);
gsl_vector *create_rand_vector(int size, double range1, double range2);
gsl_matrix *create_rand_matrix(int size1, int size2, double range1, double range2);

#endif
