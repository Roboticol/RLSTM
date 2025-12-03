#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "backprop.h"
#include "lstm.h"
#include "nutils.h"

void bp_series_lstm(LSTM *lstm, gsl_vector **series) {
	printf("beginning backpropagation...");

	// dE/dh gradient
	
}

void bp_X(BP_GATES gate, LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t3 = gsl_vector_calloc(lstm->hidden_dim);

	switch (gate) { // t1 = W * x, t2 = U * hp, t3 = b
		case FORGET:
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->wf, lstm->x, 0, t1);
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->uf, lstm->hp, 0, t2);
			gsl_blas_dcopy(lstm->bf, t3);
			break;
		case INPUT:
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->wi, lstm->x, 0, t1);
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->ui, lstm->hp, 0, t2);
			gsl_blas_dcopy(lstm->bi, t3);
			break;
		case OUTPUT:
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->wo, lstm->x, 0, t1);
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->uo, lstm->hp, 0, t2);
			gsl_blas_dcopy(lstm->bo, t3);
			break;
		case CAND:
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->wc, lstm->x, 0, t1);
			gsl_blas_dgemv(CblasNoTrans, 1, lstm->uc, lstm->hp, 0, t2);
			gsl_blas_dcopy(lstm->bc, t3);
			break;
	}

	gsl_blas_daxpy(1, t1, t2); // t2 = Wx + Uhp
	gsl_blas_daxpy(1, t2, t3); // t3 = Wx + Uhp + b
	gsl_blas_dcopy(t3, out); // out = Wx + Uhp + b

	gsl_vector_free(t1);
	gsl_vector_free(t2);
	gsl_vector_free(t3);
}

void bp_dEdh(LSTM *lstm, gsl_vector *y, gsl_vector *out) {
	gsl_vector *t = gsl_vector_calloc(lstm->output_dim);
	gsl_blas_dcopy(lstm->y, t);
	
	gsl_blas_daxpy(-1, y, t); // lstm->y = lstm->y - y (predicted - target)
	mul_vector(t, 2, t); // lstm->y = 2 * lstm->y
	gsl_blas_dgemv(CblasTrans, 1, lstm->wy, t, 0, out); // lstm->y * (Wy)T = dE/dh_t, T -> transpose
	
	gsl_vector_free(t);
}

void bp_dhdc(LSTM *lstm, gsl_vector *out) {
	gsl_vector *t = gsl_vector_calloc(lstm->hidden_dim);
	sech_vector(lstm->c, t); // sech(c)
	hdm_vector(t, t, t); // sech^2(c)

	hdm_vector(lstm->o, t, out); // out = o * sech^2(c)
		
	gsl_vector_free(t);
}

void bp_dhdo(LSTM *lstm, gsl_vector *out) {
	sigmoid_vector(lstm->c, out); // out = sigmoid(c)
}

// these functions don't take any arguments because they just use pre-existing variables inside the lstm as input!
void bp_dgdW(BP_GATES gate, LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(gate, lstm, t1); // calculate X
	sigmoid_vector(t1, t1); // t1 = sigmoid(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sigmoid(X)
	hdm_vector(t1, t2, t2); // t2 = sigmoid(X) * (1 - sigmoid(X))

	hdm_vector(lstm->x, t2, out); // out = sigmoid(X) * (1 - sigmoid(X)) * x

	gsl_vector_free(t1);
	gsl_vector_free(t2);
}

void bp_dgdU(BP_GATES gate, LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(gate, lstm, t1); // calculate X
	sigmoid_vector(t1, t1); // t1 = sigmoid(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sigmoid(X)
	hdm_vector(t1, t2, t2); // t2 = sigmoid(X) * (1 - sigmoid(X))

	hdm_vector(lstm->hp, t2, out); // out = sigmoid(X) * (1 - sigmoid(X)) * hp

	gsl_vector_free(t1);
	gsl_vector_free(t2);
}

void bp_dgdb(BP_GATES gate, LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(gate, lstm, t1); // calculate X
	sech_vector(t1, t1); // t1 = sech(X)
	hdm_vector(t1, t1, t1); // t1 = sech^2(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sech^2(X)
	hdm_vector(t1, t2, out); // t2 = sech^2(X) * (1 - sech^2(X))

	gsl_vector_free(t1);
	gsl_vector_free(t2);

}

void bp_dcadW(LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(CAND, lstm, t1); // calculate X
	sech_vector(t1, t1); // t1 = sech(X)
	hdm_vector(t1, t1, t1); // t1 = sech^2(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sech^2(X)
	hdm_vector(t1, t2, out); // t2 = sech^2(X) * (1 - sech^2(X))

	hdm_vector(lstm->x, t2, out); // out = sech^2(X) * (1 - sech^2(X)) * x

	gsl_vector_free(t1);
	gsl_vector_free(t2);
}

void bp_dcadU(LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(CAND, lstm, t1); // calculate X
	sech_vector(t1, t1); // t1 = sech(X)
	hdm_vector(t1, t1, t1); // t1 = sech^2(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sech^2(X)
	hdm_vector(t1, t2, out); // t2 = sech^2(X) * (1 - sech^2(X))

	hdm_vector(lstm->hp, t2, out); // out = sech^2(X) * (1 - sech^2(X)) * hp

	gsl_vector_free(t1);
	gsl_vector_free(t2);
}

void bp_dcadb(LSTM *lstm, gsl_vector *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(CAND, lstm, t1); // calculate X
	sech_vector(t1, t1); // t1 = sech(X)
	hdm_vector(t1, t1, t1); // t1 = sech^2(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sech^2(X)
	hdm_vector(t1, t2, out); // t2 = sech^2(X) * (1 - sech^2(X))

	gsl_vector_free(t1);
	gsl_vector_free(t2);

}

void bp_dcdf(LSTM *lstm, gsl_vector *out) {
	gsl_blas_dcopy(lstm->cp, out);
}

void bp_dcdi(LSTM *lstm, gsl_vector *out) {
	gsl_blas_dcopy(lstm->ca, out);
}

void bp_dcdca(LSTM *lstm, gsl_vector *out) {
	gsl_blas_dcopy(lstm->i, out);
}

void bp_dEdP(BP_GATES gate, BP_PARA para, LSTM *lstm, gsl_vector *out) {
	
}

void bp_dEdPo(BP_PARA para, LSTM *lstm, gsl_vector *out) {

}

void bp_dEdc(int t, LSTM **lstms, gsl_vector *out) {
	// STEP 1:
	// calculate gradient flowing from hidden state h -> cell state c at timestep t
	// dE/dct = dE/dht * dh/dct
	
}
