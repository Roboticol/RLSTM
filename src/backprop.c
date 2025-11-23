#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "lstm.h"
#include "nutils.h"

void bp_series_lstm(LSTM *lstm, gsl_vector **series) {
	printf("beginning backpropagation...");

	// dE/dh gradient
	
}

void bp_dEdh(LSTM *lstm, gsl_vector *y, gsl_vector *o) {
	gsl_vector *t = gsl_vector_calloc(lstm->output_dim);
	gsl_blas_dcopy(lstm->y, t);
	
	gsl_blas_daxpy(-1, y, t); // lstm->y = lstm->y - y (predicted - target)
	mul_vector(t, 2, t); // lstm->y = 2 * lstm->y
	gsl_blas_dgemv(CblasTrans, 1, lstm->wy, t, 0, o); // lstm->y * (Wy)T = dE/dh_t, T -> transpose
	
	gsl_vector_free(t);
}
