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
// example 1: dE/dWf = (2(y-lstm->y) * Wy) * (o * sech^2(c)) * (cp) * (sigmoid(Wfx + Ufhp + bh) * (1 - sigmoid(Wfx + Ufhp + bh)*x)
// example 2: dE/dWo = dE/dh * dh/do * do/dWo (chain rule)
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

// gradient loss wrt model parameters (W, U, and b)
// the capital P here means what parameter we're calculating with respect to, it can be W - Weight, U - recurrent kernel weights, b - bias vectors
void bp_tdEdP(BP_GATES gate, BP_PARA para, int t, LSTM *lstm, gsl_vector *out); // calculate gradient loss wrt gate parameter. only works for forget, input and candidate gates!
void bp_tdEdPo(BP_PARA para, int t, LSTM *lstm, gsl_vector *out); // calculate gradient loss wrt parameters of output gate
void bp_tdEdc(int t, LSTM_L *list, gsl_vector **series, gsl_vector *out); // calculate dEdc (gradient loss wrt cell state at timestep t)

// learning functions
void bp_lWg(BP_GATES gate, LSTM *lstm, gsl_matrix *p); // change the weight parameter of gate, i.e:
// Wf = Wf - learning rate * dE/dWf (where vector p is dE/dWf)
void bp_lUg(BP_GATES gate, LSTM *lstm, gsl_matrix *p); // change the recurrent weight parameter of gate, i.e:
// Uf = Uf - learning rate * dE/dUf (where vector p is dE/dUf)
void bp_lbg(BP_GATES gate, LSTM *lstm, gsl_vector *p); // change the bias parameter of gate, i.e:
// bf = bf - learning rate * dE/dbf (where vector p is dE/dbf)


#endif
