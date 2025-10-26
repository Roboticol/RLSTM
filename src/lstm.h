#ifndef LSTM_H
#define LSTM_H

// w = Weights matrix
// u = Recurrent weights matrix
// b = bias vector
// x = input vector
// h = hidden state vector
// c = cell state vector
// res = resultant vector
//
// In gate and other functions, which are used by the lstm and can be used by the user to test different equations i.e gate(...) and cstate_eq(...), the parameters for input have the suffix i and output have the suffix o.
//
// range1, range2 = min, max. generally used for determining range of randomly generated values
//
// read more information in: https://en.wikipedia.org/wiki/Long_short-term_memory

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
	
	// output vectorsran
	gsl_vector *h;
	gsl_vector *c;
} LSTM;

// lstm functions
LSTM* create_lstm(int input_dim, int hidden_dim); // (ONLY USE THESE FUNCTION FOR CREATING LSTMS) create lstm with all values initialized to 0;
void forward_pass_lstm(LSTM *lstm); // does a forward pass
void forward_pass_n_lstm(LSTM *lstm, gsl_vector **arr, int n); // does a forward pass on the same lstm n times. takes in an array of vectors as input, where each vector shows the change from the previous vector in a series. (arr length = n)
void free_lstm(LSTM* lstm); // delete lstm
void print_lstm(LSTM* lstm); // print lstm's contents

// randomize functions
void randomize_lstm(LSTM *lstm, double range1m, double range2m, double range1v, double range2v); // initialize LSTM with random values in a range. pre-requisite: all objects inside the struct should already be initialized.
LSTM *create_rand_lstm(int input_dim, int hidden_dim, double range1m, double range2m, double range1v, double range2v); // create LSTM with random values. This creates objects within the struct too. (ONLY RANDOMIZES WEIGHTS AND BIASES!)
LSTM *randomize_in_lstm(LSTM *lstm, double range1, double range2); // randomize lstm's inputs

// general gate function
void gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo);

// gate functions
void forget_gate_lstm(LSTM *lstm);
void input_gate_lstm(LSTM *lstm);
void output_gate_lstm(LSTM *lstm);

// other equation functions
void candidate_gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo);
void cstate_eq(gsl_vector *fi, gsl_vector *cpi, gsl_vector *ii, gsl_vector *cai, gsl_vector *co); // for cell state equation
void hstate_eq(gsl_vector *oi, gsl_vector *ci, gsl_vector *ho); // for hidden state equation
void candidate_gate_lstm(LSTM *lstm);
void cstate_eq_lstm(LSTM *lstm);
void hstate_eq_lstm(LSTM *lstm);

#endif
