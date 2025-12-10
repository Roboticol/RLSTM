#ifndef BCKPROP_H
#define BCKPROP_H

#include "lstm.h"
#include "nutils.h"

// Formulae:
// Error: E = (predicted value - true value)^2
// Output of LSTM (Not to be confused with output gate!): y = Wy * ht + by (Wy -> weight of output, ht -> hidden state of lstm at timestep t, by -> bias of output)
// dsigmoid/dx = sigmoid(x) * (1 - sigmoid(x))
// dE/dh = 2(lstm->y - y) * Wy
// dh/do = sigmoid(c)
// dh/dc = o * sech^2(c)
// dE/dW(gate) = dE/dh * dh/dc * dc/d(gate) * d(gate)/dW(gate) (only applies to forget, input/update and candidate gates).
//
// dE/dWo = dE/dh * dh/do * do/dWo
//
// dc/df = cp
// dc/di = ca (ca -> candidate cell state)
// dc/dca = i
//
// A pattern i've observed is that i, o and f gates have the same formula template. Hence we can write a general equation for them, that is:
// g = sigmoid(Wx + Uhp + b)
// hence the gradient wrt to weights and biases will be:
// dg/dW = sigmoid(X) * (1 - sigmoid(X)) * x
// where X = Wx + Uhp + b
// similarily, for the rest:
// dg/dU = sigmoid(X) * (1 - sigmoid(X)) * hp
// dg/db = sigmoid(X) * (1 - sigmoid(X))
//
// for candidate gate:
// dca/dW = sech^2(X) * x
// dca/dU = sech^2(X) * hp
// dca/db = sech^2(X)
//
// so far the formulas i've given are for only one timestep. For multiple timesteps:
// dE/dWf = summation(t = 1, T, dE/dat * xt^T) where a = Wx + Uhp + b
// dE/dc = dE/dht * dht/dct + summation(k=t+1, T, dE/dck*dck/dct)
// first term is influence from hidden term (h) and second term is influence from future cell states
// dck/dct = multiplication(j = t + 1, k, fj) where f -> forget gate output
//
// recurrence form:
// dE/dct = dE/dht * ot(sech^2(ct)) + f(t+1) * dE/dc(t+1)
//
// note: the p after the variable names denotes the previous state of the variables, i.e: cp = c(t-1) where t-> time

// an enum for the different gates, CAND = CANDIDATE 
typedef enum {INPUT, OUTPUT, FORGET, CAND} BP_GATES;

// an enum for different parameters. W - Weights, U - Recurrent Weights, b - bias vectors
typedef enum {W, U, b} BP_PARA;

// a struct for the "context" of our backpropagation algorithm, it stores values like total error, total gradient loss wrt all parameters of the lstm, etc.
typedef struct {   
	gsl_vector *dEdWf;
	gsl_vector *dEdUf;
	gsl_vector *dEdbf;

	gsl_vector *dEdWi;
	gsl_vector *dEdUi;
	gsl_vector *dEdbi;

	gsl_vector *dEdWo;
	gsl_vector *dEdUo;
	gsl_vector *dEdbo;

	gsl_vector *dEdWc;
	gsl_vector *dEdUc;
	gsl_vector *dEdbc;
} BCKPROP_CXT;

// backprop context functions
BCKPROP_CXT *bp_create_cxt(LSTM *lstm);
void bp_delete_cxt(BCKPROP_CXT *cxt);

// backpropagate an lstm along a series of vectors (backpropagation through time)
// backpropagation works like this:
// during forward pass, for each element in the series we clone the LSTM. And we store the all the calculated vectors inside that LSTM. We then move forward to the next element and keep repeating it until we reach the last element of the series.
// we also store all the losses of each timestep into a list and sum them up.
// we then start the backward pass. We go to the (n-1)th element and calculate gradients for it wrt each weight and bias
void bp_series_lstm(LSTM* lstm, gsl_vector **series, int n);

// utility functions
void bp_X(BP_GATES gate, LSTM *lstm, gsl_vector *out); // calculate X = Wx + Uhp + b
LSTM_L *bp_fwdpass(LSTM *lstm, gsl_vector **series, int n); // do a forward pass, store all the variables of the unrolled lstm in list. length of series = n

// gradient functions
void bp_dEdh(LSTM *lstm, gsl_vector *y, gsl_vector *out); // compute gradient loss wrt hidden state.
// y = actual/target output
void bp_dhdc(LSTM *lstm, gsl_vector *out); // compute gradient of hidden state wrt cell state.
void bp_dhdo(LSTM *lstm, gsl_vector *out); // compute gradient of hidden state wrt output gate vector

// these functions don't take any arguments because they just use pre-existing variables inside the lstm as input!
// compute gradients of g gate (input gate, output gate and forget gate)
// NOTE: THESE FUNCTIONS ARE NO LONGER IN USE
// void bp_dgdW(BP_GATES gate, LSTM *lstm, gsl_vector *out);
// void bp_dgdU(BP_GATES gate, LSTM *lstm, gsl_vector *out);
// void bp_dgdb(BP_GATES gate, LSTM *lstm, gsl_vector *out);
// compute gradients of candidate gate
// void bp_dcadW(LSTM *lstm, gsl_vector *out); 
// void bp_dcadU(LSTM *lstm, gsl_vector *out);
// void bp_dcadb(LSTM *lstm, gsl_vector *out);

// these functions follow the formula: dE/df = dc/df * df/dX (where X = Wx + Uhp + b)
void bp_dEdf(LSTM *lstm, gsl_vector *dEdc, gsl_vector *out); // compute gradient of cell state wrt forget vector
void bp_dEdi(LSTM *lstm, gsl_vector *dEdc, gsl_vector *out); // compute gradient of cell state wrt input gate vector
void bp_dEdca(LSTM *lstm, gsl_vector *dEdc, gsl_vector *out); // compute gradient of cell state wrt candidate gate vector

// gradient loss wrt model parameters (W, U, and b)
// the capital P here means what parameter we're calculating with respect to, it can be W - Weight, U - recurrent kernel weights, b - bias vectors
void bp_dEdP(BP_GATES gate, BP_PARA para, LSTM *lstm, gsl_matrix *out); // calculate gradient loss wrt gate parameter.
// i.e, bp_dEdP(FORGET, W, lstm, out); is equivalent to writing dE/dWf, which is the gradient loss wrt weight of forward gate
void bp_tdEdc(int t, LSTM_L *list, gsl_vector **series, gsl_vector *out); // calculate dEdc (gradient loss wrt cell state at timestep t)

// learning functions
void bp_lWg(BP_GATES gate, LSTM *lstm, gsl_matrix *p); // change the weight parameter of gate, i.e:
// Wf = Wf - learning rate * dE/dWf (where vector p is dE/dWf)
void bp_lUg(BP_GATES gate, LSTM *lstm, gsl_matrix *p); // change the recurrent weight parameter of gate, i.e:
// Uf = Uf - learning rate * dE/dUf (where vector p is dE/dUf)
void bp_lbg(BP_GATES gate, LSTM *lstm, gsl_vector *p); // change the bias parameter of gate, i.e:
// bf = bf - learning rate * dE/dbf (where vector p is dE/dbf)


#endif
