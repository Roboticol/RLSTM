#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nutils.h"

void init_utils() {
	// initializes the randomizer
	srand(time(NULL));
}

double sigmoid(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

void tanh_vector(gsl_vector *v, gsl_vector *r) {
	int size = v->size;
	
	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, tanh(gsl_vector_get(v, i)));
	}
}

void sech_vector(gsl_vector *v, gsl_vector *r) {
	int size = v->size;
	
	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, 1/cosh(gsl_vector_get(v, i)));
	}
}

void concatenate_vector(gsl_vector *a, gsl_vector *b, gsl_vector *r) {
	int asize = a->size;
	int bsize = b->size;

	int size = asize + bsize;

	for (int i = 0; i < size; i++) {
		if(i < asize) gsl_vector_set(r, i, gsl_vector_get(a, i));
		else gsl_vector_set(r, i, gsl_vector_get(b, i-asize));
	}	
}

void sigmoid_vector(gsl_vector *v, gsl_vector *r) {
	int size = v->size;
	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, sigmoid(gsl_vector_get(v, i)));
	}
}

void hdm_vector(gsl_vector *a, gsl_vector *b, gsl_vector *r) {
	int size = a->size;

	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, gsl_vector_get(a, i) * gsl_vector_get(b, i));
	}
}
		
void print_vector(gsl_vector *v, char *s) {
	printf("%s", s);
	for (int i = 0; i < (int)v->size; i++) {
		printf("%.2lf ", gsl_vector_get(v, i));
	}
	printf("\n");
}

void print_matrix(gsl_matrix *m, char *s) {
	printf("%s\n", s);
	for (int i = 0; i < (int)m->size1; i++) {
		for (int j = 0; j < (int)m->size2; j++) {
			printf("%.2lf ", gsl_matrix_get(m, i, j));
		}
		printf("\n");
	}
	printf("\n");
}

double random_double(double range1, double range2) {
	return range1 + ((double)rand() / (RAND_MAX / (range2 - range1)));
}

void randomize_vector(gsl_vector *x, double range1, double range2) {
	int size = x->size;

	for (int i = 0; i < size; i++) {
		gsl_vector_set(x, i, random_double(range1, range2));
	}
}

void randomize_matrix(gsl_matrix *x, double range1, double range2) {
	int rows = x->size1;
	int cols = x->size2;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			gsl_matrix_set(x, i, j, random_double(range1, range2));
		}
	}
}

gsl_vector *create_rand_vector(int size, double range1, double range2) {
	gsl_vector *v = gsl_vector_alloc(size);
	randomize_vector(v, range1, range2);
	return v;
}

gsl_matrix *create_rand_matrix(int size1, int size2, double range1, double range2) {
	gsl_matrix *m = gsl_matrix_alloc(size1, size2);
	randomize_matrix(m, range1, range2);
	return m;
}

gsl_vector **series_vectors(int size, int n, double range1i, double range2i, double range1v, double range2v) {
	gsl_vector **vl = (gsl_vector **)malloc(n * sizeof(gsl_vector*)); // initialize an array of pointers, that point to vectors.
	gsl_vector *v = gsl_vector_calloc(size); // initialize first vector
	randomize_vector(v, range1i, range2i);	

	vl[0] = v;
	
	for (int i = 1; i < n; i++) {
		vl[i] = gsl_vector_calloc(size);	
		for (int j = 0; j < size; j++) {
			double delta = random_double(range1v, range2v); // amount the vector changes by
			gsl_vector_set(vl[i], j, gsl_vector_get(vl[i-1], j) + delta);
		}
	}

	return vl;
}

void free_series_vectors(gsl_vector **v, int n) {
	for (int i = 0; i < n; i++) {
		gsl_vector_free(v[i]);
	}
}

void print_series_vectors(gsl_vector **v, int n, char *s) {
	printf("%s\n", s);
	for (int i = 0; i < n; i++) {
		printf("%d: ", i);
		print_vector(v[i], "");
	}
}

double mse(double a, double b) {
	return pow((a - b), 2);
}

double mse_vector(gsl_vector *a, gsl_vector *b) {
	double res = 0;
	int size = a->size;

	for (int i = 0; i < size; i++) {
		// perform mse on different values of vectors and add them to result variable res
		res += mse(gsl_vector_get(a, i), gsl_vector_get(b, i));
	}
	return res/((double)size);
}

void mul_vector(gsl_vector *a, double c, gsl_vector *r) {
	int size = a->size;

	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, gsl_vector_get(a, i) * c);
	}
}

void add_vector(double b, gsl_vector *a, double c, gsl_vector *r) {
	int size = a->size;

	for (int i = 0; i < size; i++) {
		gsl_vector_set(r, i, (b * gsl_vector_get(a, i)) + c);
	}
}

void mul_matrix(gsl_matrix *a, double c, gsl_matrix *r) {
	int size1 = a->size1;
	int size2 = a->size2;

	for (int y = 0; y < size1; y++) {
		for (int x = 0; x < size2; x++) {
			gsl_matrix_set(r, y, x, gsl_matrix_get(a, y, x) * c);
		}
	}
}

void add_matrix(gsl_matrix *a, double b, gsl_matrix *c, double d, double e, gsl_matrix *r) {
	// formula: a * b + c * d + e = r. 
	// a and c -> matrices
	// b, d, e -> constants
	int size1 = a->size1;
	int size2 = a->size2;

	for (int y = 0; y < size1; y++) {
		for (int x = 0; x < size2; x++) {
			gsl_matrix_set(r, y, x, (gsl_matrix_get(a, y, x) * b) + (gsl_matrix_get(c, y, x) * d) + e);
		}
	}
}

gsl_matrix *convert_vtm(CBLAS_TRANSPOSE_t trans, gsl_vector *v) {
	// converts vector of dimension n into a matrix of dimension n * 1 or 1 * n
	gsl_matrix *m; 

	switch (trans) {
		case CblasNoTrans:
			m = gsl_matrix_calloc(v->size, 1);
			break;
		case CblasTrans:
			m = gsl_matrix_calloc(1, v->size);
			break;
		default:
			printf("Invalid option\n");
			break;
	}

	for (int i = 0; i < v->size; i++) {
		switch (trans) {
			case CblasNoTrans:
				gsl_matrix_set(m, i, 1, gsl_vector_get(v, i));
				break;
			case CblasTrans:
				gsl_matrix_set(m, 1, i, gsl_vector_get(v, i));
				break;
			default:
				break;
		}
	}

	return m;
}
