#ifndef LSTM_
#define LSTM_

// w = Weights matrix
// u = Recurrent weights matrix
// b = bias vector
// x = input vector
// h = hidden state vector
// c = cell state vector
// res = resultant vector
//
// f = forget gate vector
// i = input/update gate vector
// o = output gate vector
// c1 = cell input activation vector
//
// i = input
// o = output

void testfunc();

void forget_gate(gsl_matrix *wi, gsl_matrix *ui, gsl_vector *bi, gsl_vector *xi, gsl_vector *hi, gsl_vector *fo);

#endif
