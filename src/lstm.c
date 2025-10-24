#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "nutils.h"
#include <gsl/gsl_blas.h>

void forget_gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo) {
	// Formula used: sigmoid(wi * xi + ui * hi + bi)

	int input_dim = xi->size;
	int hidden_dim = hi->size;

	int concatsize = input_dim + hidden_dim;

	// Vector initialization
	gsl_vector *wixi = gsl_vector_calloc(hidden_dim);
	gsl_vector *uihi = gsl_vector_calloc(hidden_dim);

	// Matrix multiplication
	gsl_blas_dgemv(CblasNoTrans, 1, wi, xi, 0, wixi);
	gsl_blas_dgemv(CblasNoTrans, 1, ui, hi, 0, uihi);

	// Vector addition
	gsl_blas_daxpy(1, wixi, uihi);
	gsl_blas_daxpy(1, bi, uihi);

	// Final sigmoid result
	sigmoid_vector(uihi, uihi);
	gsl_blas_dcopy(uihi, fo);

	// Memory safety steps
	gsl_vector_free(wixi);
	gsl_vector_free(uihi);
}

void testfunc() {
	printf("Running LSTM!\n");
	printf("Sigmoid test: s(0.65) = %lf\n", sigmoid(0.65));
}
