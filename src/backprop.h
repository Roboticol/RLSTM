#ifndef BCKPROP_H
#define BCKPROP_H

#include "nutils.h"

// Notes taken from: https://www.geeksforgeeks.org/dsa/lstm-derivation-of-back-propagation-through-time/
// gradient of loss wrt output gate
// dE/do = dE/dh * tanh(c)
//
// gradient of loss wrt cell state
// dE/dc = dE/dh * o * (1-tanh^2(c))
//
// gradient of loss wrt input gate and candidate cell state
// dE/di = dE/dc * g
// dE/dg = dE/dc * i
//
// gradient of loss wrt forget gate
// dE/df = dE/dc * cp (cp -> previous cell state)
//
// gradient of loss wrt previous cell state
// dE/dcp = dE/dc * f
//
// gradients for output gate weights
// dE/dWxo = dE/do * o(1 - o) * x
// dE/dWho = dE/do * o(1 - o) * hp
// dE/dbo = dE/do * o(1 - o)
//
// lstm->y = lstm's predicted output
// y = actual/target output


// backpropagate an lstm along a series of vectors
void bp_series_lstm(LSTM* lstm, gsl_vector **series);

// gradient functions
void bp_dEdh(LSTM *lstm, gsl_vector *y, gsl_vector *o); // compute gradient loss wrt hidden state. dE/dh.
// y = actual/target output


#endif
