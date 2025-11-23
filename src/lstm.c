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

	int hidden_dim = hi->size;

	// Vector initialization
	gsl_vector *wixi = gsl_vector_calloc(hidden_dim);
	gsl_vector *uihi = gsl_vector_calloc(hidden_dim);

	// Matrix multiplication
	gsl_blas_dgemv(CblasNoTrans, 1, wi, xi, 0, wixi);
	gsl_blas_dgemv(CblasNoTrans, 1, ui, hi, 0, uihi);

	// Vector addition
	gsl_blas_daxpy(1, wixi, uihi); // adding to uihi because it will be deleted anyway
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
	LSTM *lstm = (LSTM *)malloc(sizeof(LSTM));
	if (lstm == NULL) printf("ERROR: FAILED TO ALLOCATE LSTM STRUCT!\n");

	// matrices
	lstm->wf = gsl_matrix_calloc(hidden_dim, input_dim);
	lstm->wi = gsl_matrix_calloc(hidden_dim, input_dim);
	lstm->wo = gsl_matrix_calloc(hidden_dim, input_dim);
	lstm->wc = gsl_matrix_calloc(hidden_dim, input_dim);

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

	return lstm;
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

void randomize_lstm(LSTM *lstm, double range1m, double range2m, double range1v, double range2v) {
	// weight matrices
	randomize_matrix(lstm->wf, range1m, range2m);
	randomize_matrix(lstm->wi, range1m, range2m);
	randomize_matrix(lstm->wo, range1m, range2m);
	randomize_matrix(lstm->wc, range1m, range2m);

	randomize_matrix(lstm->uf, range1m, range2m);
	randomize_matrix(lstm->ui, range1m, range2m);
	randomize_matrix(lstm->uo, range1m, range2m);
	randomize_matrix(lstm->uc, range1m, range2m);

	// bias vectors
	randomize_vector(lstm->bf, range1v, range2v);
	randomize_vector(lstm->bi, range1v, range2v);
	randomize_vector(lstm->bo, range1v, range2v);
	randomize_vector(lstm->bc, range1v, range2v);

	// input vectors
	// randomize_vector(lstm->x, range1v, range2v);
	// randomize_vector(lstm->hp, range1v, range2v); // h(t-1) vector
	// randomize_vector(lstm->cp, range1v, range2v);
	
	// intermediate vectors (these are used within the LSTM)
	// randomize_vector(lstm->f, range1v, range2v);
	// randomize_vector(lstm->i, range1v, range2v);
	// randomize_vector(lstm->o, range1v, range2v);
	// randomize_vector(lstm->ca, range1v, range2v); // candidate vector
	
	// output vectors
	// randomize_vector(lstm->h, range1v, range2v);
	// randomize_vector(lstm->c, range1v, range2v);
}

LSTM *create_rand_lstm(int input_dim, int hidden_dim, double range1m, double range2m, double range1v, double range2v) {
	LSTM *lstm = create_lstm(input_dim, hidden_dim);
	randomize_lstm(lstm, range1m, range2m, range1v, range2v);
	return lstm;
}

void randomize_in_lstm(LSTM *lstm, double range1, double range2) {
	// randomizes all input vectors of lstm
	randomize_vector(lstm->x, range1, range2);
	randomize_vector(lstm->hp, range1, range2);
	randomize_vector(lstm->cp, range1, range2);
}

void print_lstm(LSTM* lstm) {	
	// matrices
	printf("====Matrices====\n");
	printf("--Weights Matrices--\n");
	print_matrix(lstm->wf, "wf: ");
	print_matrix(lstm->wi, "wi: ");
	print_matrix(lstm->wo, "wo: ");
	print_matrix(lstm->wc, "wc: ");

	printf("--Hidden State Weights Matrices--\n");
	print_matrix(lstm->uf, "uf: ");
	print_matrix(lstm->ui, "ui: ");
	print_matrix(lstm->uo, "uo: ");
	print_matrix(lstm->uc, "uc: ");

	// bias vectors
	printf("====Vectors====\n");
	printf("--Bias Vectors--\n");
	print_vector(lstm->bf, "bf: ");
	print_vector(lstm->bi, "bi: ");
	print_vector(lstm->bo, "bo: ");
	print_vector(lstm->bc, "bc: ");

	// input vectors
	printf("--Input Vectors--\n");
	print_vector(lstm->x, "x: ");
	print_vector(lstm->hp, "hp: ");
	print_vector(lstm->cp, "cp: ");

	// intermediate vectors
	printf("--Intermediate Vectors--\n");
	print_vector(lstm->f, "f: ");
	print_vector(lstm->i, "i: ");
	print_vector(lstm->o, "o: ");
	print_vector(lstm->ca, "ca: ");

	// output vectors
	printf("--Output Vectors--\n");
	print_vector(lstm->h, "h: ");
	print_vector(lstm->c, "c: ");
}


void forget_gate_lstm(LSTM *lstm) {
	gate(lstm->wf, lstm->uf, lstm->bf, lstm->x, lstm->hp, lstm->f);
}

void input_gate_lstm(LSTM *lstm) {
	gate(lstm->wi, lstm->ui, lstm->bi, lstm->x, lstm->hp, lstm->i);
}

void output_gate_lstm(LSTM *lstm) {
	gate(lstm->wo, lstm->uo, lstm->bo, lstm->x, lstm->hp, lstm->o);
}

// cell input activation vector gate
void candidate_gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo) {
	// Formula used: tanh(wi * xi + ui * hi + bi)

	int input_dim = xi->size;
	int hidden_dim = hi->size;

	// Vector initialization
	gsl_vector *wixi = gsl_vector_calloc(hidden_dim);
	gsl_vector *uihi = gsl_vector_calloc(hidden_dim);

	// Matrix multiplication
	gsl_blas_dgemv(CblasNoTrans, 1, wi, xi, 0, wixi);
	gsl_blas_dgemv(CblasNoTrans, 1, ui, hi, 0, uihi);

	// Vector addition
	gsl_blas_daxpy(1, wixi, uihi); // adding to uihi because it will be deleted anyway
	gsl_blas_daxpy(1, bi, uihi);

	// Final tanh result
	tanh_vector(uihi, uihi);
	gsl_blas_dcopy(uihi, fo);

	// Memory safety steps
	gsl_vector_free(wixi);
	gsl_vector_free(uihi);
}

void candidate_gate_lstm(LSTM *lstm) {
	candidate_gate(lstm->wc, lstm->uc, lstm->bc, lstm->x, lstm->hp, lstm->ca);
}

void cstate_eq(gsl_vector *fi, gsl_vector *cpi, gsl_vector *ii, gsl_vector *cai, gsl_vector *co) {
	// formula used: fi * cpi + ii * cai ( * = hadamard product)
	// initialize vectors
	gsl_vector *hdm1 = gsl_vector_calloc(fi->size);	
	gsl_vector *hdm2 = gsl_vector_calloc(fi->size);	
	
	// hadarmard product of vectors
	hdm_vector(fi, cpi, hdm1);
	hdm_vector(ii, cai, hdm2);

	// addition of vectors
	gsl_blas_daxpy(1, hdm1, hdm2);

	// copying into resultant vector
	gsl_blas_dcopy(hdm2, co);

	// memory safety steps
	gsl_vector_free(hdm1);
	gsl_vector_free(hdm2);
}

void hstate_eq(gsl_vector *oi, gsl_vector *ci, gsl_vector *ho) {
	// formula used: oi * tanh(ci) ( * = hadamard product)
	// initialize vectors
	gsl_vector *v = gsl_vector_calloc(oi->size);
	gsl_vector *s = gsl_vector_calloc(oi->size);

	// tanh of vector
	tanh_vector(ci, s);

	// getting product
	hdm_vector(oi, s, v);
	
	// copying over to resultant vector
	gsl_blas_dcopy(v, ho);

	// memory safety steps
	gsl_vector_free(v);	
	gsl_vector_free(s);
}

void cstate_eq_lstm(LSTM *lstm) {
	cstate_eq(lstm->f, lstm->cp, lstm->i, lstm->ca, lstm->c);
}

void hstate_eq_lstm(LSTM *lstm) {
	hstate_eq(lstm->o, lstm->c, lstm->h);
}

void forward_pass_lstm(LSTM *lstm) {
	// calculate all equations
	forget_gate_lstm(lstm);
	input_gate_lstm(lstm);
	output_gate_lstm(lstm);
	candidate_gate_lstm(lstm);

	cstate_eq_lstm(lstm);
	hstate_eq_lstm(lstm);
}

void forward_pass_n_lstm(LSTM *lstm, gsl_vector **arr, int n) {
	for (int i = 0; i < n; i++) {
		gsl_blas_dcopy(arr[i], lstm->x);
		forward_pass_lstm(lstm);
		gsl_blas_dcopy(lstm->h, lstm->hp);
		gsl_blas_dcopy(lstm->c, lstm->cp);
	}	
}

void input_vector_lstm(LSTM* lstm, gsl_vector *v) {
	gsl_blas_dcopy(v, lstm->x);
}
