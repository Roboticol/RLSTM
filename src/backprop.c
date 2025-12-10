#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "backprop.h"
#include "lstm.h"
#include "nutils.h"

static double learning_rate = 0.001;

void bp_series_lstm(LSTM *lstm, gsl_vector **series, int n) {
	printf("beginning backpropagation...");

	// dE/dh gradient
	LSTM_L *list = bp_fwdpass(lstm, series, n);
	BCKPROP_CXT *context = bp_create_cxt(lstm);

	for (int i = 0; i < n; i++) {
		// formulas in backprop.h
		gsl_vector *dEdc = gsl_vector_calloc(lstm->hidden_dim);
		bp_tdEdc(i, list, series, dEdc);

		gsl_vector *dcdf = gsl_vector_calloc(lstm->hidden_dim);
		bp_dcdf(lstm, dcdf);
		
		gsl_vector *dfdW = gsl_vector_calloc(lstm->hidden_dim);
		bp_dgdW(FORGET, lstm, dfdW);

		// calculate the final dEdWft, gradient loss w.r.t weight of forget gate.
		gsl_vector *dEdWf = gsl_vector_calloc(lstm->hidden_dim); 
		hdm_vector(dEdc, dcdf, dEdWf);
		hdm_vector(dEdWf, dfdW, dEdWf);

		// add up all the gradients
		gsl_vector_daxpy(1, dEdWf, dEdWf);
		add_matrix(dEdWf, 1, context->dEdWf, 1, 0, context->dEdWf) // context->dEdWf += dEdWf

	}

	lstml_deletex(list);
	bp_delete_cxt(context);
}

LSTM_L *bp_fwdpass(LSTM *lstm, gsl_vector **series, int n) {
	LSTM_L *l = lstml_create(); // create lstm list (unrolled lstm)

	for (int i = 0; i < n; i++) {
		if (i > 0) {
			// copy outputs from last lstm output
			gsl_blas_dcopy(lstm->c, lstm->cp);
			gsl_blas_dcopy(lstm->h, lstm->hp);
		}

		input_vector_lstm(lstm, series[i]); // input series data at index into lstm
		forward_pass_lstm(lstm); // forward pass lstm

		LSTM *clone = clone_lstm(lstm); // clone lstm and append it to list
		lstml_append(l, clone);
	}

	return l;
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
void bp_dgdW(BP_GATES gate, LSTM *lstm, gsl_matrix *out) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector *t2 = gsl_vector_calloc(lstm->hidden_dim);

	bp_X(gate, lstm, t1); // calculate X
	sigmoid_vector(t1, t1); // t1 = sigmoid(X)
	add_vector(-1, t1, 1, t2); // t2 = 1 - sigmoid(X)
	hdm_vector(t1, t2, t2); // t2 = sigmoid(X) * (1 - sigmoid(X))

	gsl_matrix *x_matrix = convert_vtm(CblasTrans, lstm->x); // x as a tranposed matrix.
	gsl_matrix *t_matrix = convert_vtm(CblasNoTrans, t2); // t2 as a tranposed matrix.
	// we perform matrix multiplication: t_matrix * x_matrix to get our final matrix result
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 0, t_matrix, x_matrix, 0, out); // out = sigmoid(X) * (1 - sigmoid(X)) * x

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

void bp_tdEdc(int t, LSTM_L *list, gsl_vector **series, gsl_vector *out) {
	int hidden_dim = list->data[0]->hidden_dim;
	gsl_vector *res = gsl_vector_calloc(hidden_dim);

	int i2 = 0;

	for (int i = t; i < list->size; i++) {
		// STEP 1:
		// calculate gradient flowing from hidden state h -> cell state c at timestep t
		// dE/dct = dE/dht * dh/dct

		gsl_vector *t1 = gsl_vector_calloc(hidden_dim);
		gsl_vector *t2 = gsl_vector_calloc(hidden_dim);
		bp_dEdh(lstml_get(list, i), series[i], t1); // t1 = dEt/dht
		bp_dhdc(lstml_get(list, i), t2); // t2 = dht/dct
		hdm_vector(t1, t2, t1); // t1 = dEt/dht * dht/dct

		// STEP 2:
		// calculate gradient flowing from future cell states c(t+x) -> current cell state c(t)
		for (int j = 0; j < i2; j++) {
			hdm_vector(t1, lstml_get(list, i + j + 1)->f, t1);
		}

		gsl_blas_daxpy(1, t1, res);

		gsl_vector_free(t1);
		gsl_vector_free(t2);

		i2++;
	}

	gsl_blas_dcopy(res, out);
	gsl_vector_free(res);
}

void bp_lWg(BP_GATES gate, LSTM *lstm, gsl_matrix *p) {
	gsl_matrix *t1 = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	gsl_matrix **t2;
	switch(gate) {
		case INPUT:
		t2 = &(lstm->wi);
		break;
		case OUTPUT:
		t2 = &(lstm->wo);
		break;
		case FORGET:
		t2 = &(lstm->wf);
		break;
		case CAND:
		t2 = &(lstm->wc);
		break;
	}

	add_matrix(*t2, 1, p, -learning_rate, 0, *t2);
}

void bp_lUg(BP_GATES gate, LSTM *lstm, gsl_matrix *p) {
	gsl_matrix *t1 = gsl_matrix_calloc(lstm->hidden_dim, lstm->hidden_dim);
	gsl_matrix **t2;
	switch(gate) {
		case INPUT:
		t2 = &(lstm->ui);
		break;
		case OUTPUT:
		t2 = &(lstm->uo);
		break;
		case FORGET:
		t2 = &(lstm->uf);
		break;
		case CAND:
		t2 = &(lstm->uc);
		break;
	}

	add_matrix(*t2, 1, p, -learning_rate, 0, *t2);
}

void bp_lbg(BP_GATES gate, LSTM *lstm, gsl_vector *p) {
	gsl_vector *t1 = gsl_vector_calloc(lstm->hidden_dim);
	gsl_vector **t2;
	switch(gate) {
		case INPUT:
		t2 = &(lstm->bi);
		break;
		case OUTPUT:
		t2 = &(lstm->bo);
		break;
		case FORGET:
		t2 = &(lstm->bf);
		break;
		case CAND:
		t2 = &(lstm->bc);
		break;
	}

	gsl_blas_daxpy(-learning_rate, p, *t2);
}

BCKPROP_CXT *bp_create_cxt(LSTM *lstm) {
	// allocate different matrix and vector gradients
	BCKPROP_CXT *backprop_context = (BCKPROP_CXT *)malloc(BCKPROP_CXT);

	backprop_context->dEdWf = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdUf = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdbf = gsl_vector_calloc(lstm->hidden_dim);

	backprop_context->dEdWi = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdUi = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdbi = gsl_vector_calloc(lstm->hidden_dim);

	backprop_context->dEdWo = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdUo = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdbo = gsl_vector_calloc(lstm->hidden_dim);

	backprop_context->dEdWc = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdUc = gsl_matrix_calloc(lstm->hidden_dim, lstm->input_dim);
	backprop_context->dEdbc = gsl_vector_calloc(lstm->hidden_dim);

	return backprop_context;
}

void bp_delete_cxt(BCKPROP_CXT *cxt) {
	// free all resources from backprop context
	gsl_matrix_free(cxt->dEdWf);
	gsl_matrix_free(cxt->dEdUf);
	gsl_matrix_free(cxt->dEdbf);

	gsl_matrix_free(cxt->dEdWi);
	gsl_matrix_free(cxt->dEdUi);
	gsl_matrix_free(cxt->dEdbi);

	gsl_matrix_free(cxt->dEdWo);
	gsl_matrix_free(cxt->dEdUo);
	gsl_matrix_free(cxt->dEdbo);

	gsl_matrix_free(cxt->dEdWc);
	gsl_matrix_free(cxt->dEdUc);
	gsl_matrix_free(cxt->dEdbc);
	
	free(cxt);
}