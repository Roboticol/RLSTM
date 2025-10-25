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
// read more information in: https://en.wikipedia.org/wiki/Long_short-term_memory

// for testing
void testfunc();

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

// lstm functions
LSTM* create_lstm(); // (ONLY USE THESE FUNCTION FOR CREATING LSTMS) create lstm with all values initialized to 0;
void randomize_lstm(); // initialize LSTM with random values
LSTM* create_rand_lstm(); // create LSTM with random values;
void free_lstm(LSTM* lstm); // delete lstm

// general gate function
void gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo);

// gate functions
void forget_gate(LSTM *lstm);
void input_gate(LSTM *lstm);
void output_gate(LSTM *lstm);
void candidate_gate(LSTM *lstm);

#endif
