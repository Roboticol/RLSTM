#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "nutils.h"

// all variable notation in this struct is taken from https://en.wikipedia.org/wiki/Long_short-term_memory
typedef struct {
	// weight matrices
	gsl_matrix *wf;
	gsl_matrix *wi;
	gsl_matrix *wo;
	gsl_matrix *wc;

	gsl_matrix *uf;
	gsl_matrix *ui;
	gsl_matrix *uo;
	gsl_matrix *uc;

	// bias vectors
	gsl_vector *bf;
	gsl_vector *bi;
	gsl_vector *bo;
	gsl_vector *bc;

	// input vectors
	gsl_vector *x;
	gsl_vector *hp; // h(t-1) vector
	gsl_vector *cp;
	
	// intermediate vectors (these are used within the LSTM)
	gsl_vector *f;
	gsl_vector *i;
	gsl_vector *o;
	gsl_vector *ca; // candidate vector
	
	// output vectors
	gsl_vector *h;
	gsl_vector *c;
} LSTM;

void gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo) {
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

// this function initializes matrices and vectors
LSTM *create_lstm(int input_dim, int hidden_dim) {
	// allocate lstm to heap
	LSTM *lstm = malloc(sizeof(LSTM));

	// matrices
	lstm->wf = gsl_matrix_calloc(input_dim, hidden_dim);
	lstm->wi = gsl_matrix_calloc(input_dim, hidden_dim);
	lstm->wo = gsl_matrix_calloc(input_dim, hidden_dim);
	lstm->wc = gsl_matrix_calloc(input_dim, hidden_dim);

	lstm->uf = gsl_matrix_calloc(hidden_dim, hidden_dim);
	lstm->ui = gsl_matrix_calloc(hidden_dim, hidden_dim);
	lstm->uo = gsl_matrix_calloc(hidden_dim, hidden_dim);
	lstm->uc = gsl_matrix_calloc(hidden_dim, hidden_dim);

	// bias vectors
	lstm->bf = gsl_vector_calloc(hidden_dim);
	lstm->bi = gsl_vector_calloc(hidden_dim);
	lstm->bo = gsl_vector_calloc(hidden_dim);
	lstm->bc = gsl_vector_calloc(hidden_dim);

	// input vectors
	lstm->x = gsl_vector_calloc(input_dim);
	lstm->hp = gsl_vector_calloc(hidden_dim);
	lstm->cp = gsl_vector_calloc(hidden_dim);

	// intermediate vectors
	lstm->f = gsl_vector_calloc(hidden_dim);
	lstm->i = gsl_vector_calloc(hidden_dim);
	lstm->o = gsl_vector_calloc(hidden_dim);
	lstm->ca = gsl_vector_calloc(hidden_dim);

	// output vectors
	lstm->h = gsl_vector_calloc(hidden_dim);
	lstm->c = gsl_vector_calloc(hidden_dim);
}

void free_lstm(LSTM* lstm) {	
	// matrices
	gsl_matrix_free(lstm->wf);
	gsl_matrix_free(lstm->wi);
	gsl_matrix_free(lstm->wo);
	gsl_matrix_free(lstm->wc);

	gsl_matrix_free(lstm->uf);
	gsl_matrix_free(lstm->ui);
	gsl_matrix_free(lstm->uo);
	gsl_matrix_free(lstm->uc);

	// bias vectors
	gsl_vector_free(lstm->bf);
	gsl_vector_free(lstm->bi);
	gsl_vector_free(lstm->bo);
	gsl_vector_free(lstm->bc);

	// input vectors
	gsl_vector_free(lstm->x);
	gsl_vector_free(lstm->hp);
	gsl_vector_free(lstm->cp);

	// intermediate vectors
	gsl_vector_free(lstm->f);
	gsl_vector_free(lstm->i);
	gsl_vector_free(lstm->o);
	gsl_vector_free(lstm->ca);

	// output vectors
	gsl_vector_free(lstm->h);
	gsl_vector_free(lstm->c);

	// free lstm struct
	free(lstm);
}

void randomize_lstm(LSTM *lstm) {

}

void forget_gate(LSTM *lstm) {
	gate(lstm->wf, lstm->uf, lstm->bf, lstm->x, lstm->hp, lstm->f);
}

void input_gate(LSTM *lstm) {
	gate(lstm->wi, lstm->ui, lstm->bi, lstm->x, lstm->hp, lstm->i);
}

void output_gate(LSTM *lstm) {
	gate(lstm->wo, lstm->uo, lstm->bo, lstm->x, lstm->hp, lstm->o);
}

// cell input activation vector gate
void candidate_gate(LSTM *lstm) {
	gate(lstm->wc, lstm->uc, lstm->bc, lstm->x, lstm->hp, lstm->ca);
}

void testfunc() {
	printf("Running LSTM!\n");
	printf("Sigmoid test: s(0.65) = %lf\n", sigmoid(0.65));
}
