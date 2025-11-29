#ifndef BCKPROP_H
#define BCKPROP_H

#include "lstm.h"
#include "nutils.h"

// Formulae:
// dsigmoid/dx = sigmoid(x) * (1 - sigmoid(x))
// dE/dh = 2(lstm->y - y) * Wy
// dh/do = sigmoid(c)
// dh/dc = o * sech^2(c)
// dE/dW(gate) = dE/dh * dh/dc * dc/d(gate) * d(gate)/dW(gate) (only applies to forget, input/update and candidate gates).
// i.e: dE/dWf = (2(y-lstm->y) * Wy) * (o * sech^2(c)) * (cp) * (sigmoid(Wfx + Ufhp + bh) * (1 - sigmoid(Wfx + Ufhp + bh)*x)
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
// note: the p after the variable names denotes the previous state of the variables, i.e: cp = c(t-1) where t-> time

// an enum for the different gates, CAND = CANDIDATE 
typedef enum {INPUT, OUTPUT, FORGET, CAND} BP_GATES;

// backpropagate an lstm along a series of vectors
void bp_series_lstm(LSTM* lstm, gsl_vector **series);

// utility functions
void bp_X(BP_GATES gate, LSTM *lstm, gsl_vector *out); // calculate X = Wx + Uhp + b

// gradient functions
void bp_dEdh(LSTM *lstm, gsl_vector *y, gsl_vector *out); // compute gradient loss wrt hidden state.
// y = actual/target output
void bp_dhdc(LSTM *lstm, gsl_vector *out); // compute gradient of hidden state wrt cell state.
void bp_dhdo(LSTM *lstm, gsl_vector *out); // compute gradient of hidden state wrt output gate vector

// these functions don't take any arguments because they just use pre-existing variables inside the lstm as input!
// compute gradients of g gate (input gate, output gate and forget gate)
void bp_dgdW(BP_GATES gate, LSTM *lstm, gsl_vector *out);
void bp_dgdU(BP_GATES gate, LSTM *lstm, gsl_vector *out);
void bp_dgdb(BP_GATES gate, LSTM *lstm, gsl_vector *out);
// compute gradients of candidate gate
void bp_dcadW(LSTM *lstm, gsl_vector *out); 
void bp_dcadU(LSTM *lstm, gsl_vector *out);
void bp_dcadb(LSTM *lstm, gsl_vector *out);

void bp_dcdf(LSTM *lstm, gsl_vector *out); // compute gradient of cell state wrt forget get vector
void bp_dcdi(LSTM *lstm, gsl_vector *out); // compute gradient of cell state wrt input gate vector
void bp_dcdca(LSTM *lstm, gsl_vector *out); // compute gradient of cell state wrt candidate gate vector


#endif
