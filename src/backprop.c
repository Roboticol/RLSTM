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
	gsl_blas_daxpy(-1, y, lstm->y); // lstm->y = lstm->y - y (predicted - target)
	mul_vector(lstm->y, 2, lstm->y); // lstm->y = 2 * lstm->y
	printf("aaaaaaaa, %ld, %ld, %ld\n", lstm->y->size, lstm->wy->size1, lstm->wy->size2);
	gsl_blas_dgemv(CblasTrans, 1, lstm->wy, lstm->y, 0, o); // lstm->y * Wy = dE/dh_t
}
