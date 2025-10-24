#include <stdio.h>
#include <math.h>
#include "nutils.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

double sigmoid(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

gsl_vector *concatenate_vector(gsl_vector *a, gsl_vector *b) {
	int asize = a->size;
	int bsize = b->size;

	int size = asize + bsize;

	gsl_vector *v = gsl_vector_calloc(size);

	for (int i = 0; i < size; i++) {
		if(i < asize) gsl_vector_set(v, i, gsl_vector_get(a, i));
		else gsl_vector_set(v, i, gsl_vector_get(b, i-asize));
	}
	
	return v;
}

void *sigmoid_vector(gsl_vector *v, gsl_vector *r) {
	int size = v->size;
	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, sigmoid(gsl_vector_get(v, i)));
	}
}

void print_vector(gsl_vector *v) {
	for (int i = 0; i < v->size; i++) {
		printf("%lf ", gsl_vector_get(v, i));
	}
	printf("\n");
}

void print_matrix(gsl_matrix *m) {
	for (int i = 0; i < m->size1; i++) {
		for (int j = 0; j < m->size2; j++) {
			printf("%lf ", gsl_matrix_get(m, i, j));
		}
		printf("\n");
	}
	printf("\n");
}
