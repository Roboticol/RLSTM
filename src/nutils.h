#ifndef NUTILS_
#define NUTILS_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846
#define EULER_NUMBER_L 2.71828182845904523536

double sigmoid(double n);
gsl_vector *concatenate_vector(gsl_vector *a, gsl_vector *b);
void *sigmoid_vector(gsl_vector *v, gsl_vector *r);

void print_vector(gsl_vector *v);
void print_matrix(gsl_matrix *m);

#endif
