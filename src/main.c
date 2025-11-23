// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"

int main() {
	init_utils();

	int inp_dim = 1;
	int hidden_dim = 2;
	LSTM* l = create_rand_lstm(inp_dim,hidden_dim,-1,1,-10,10);
	randomize_in_lstm(l, -1,1);

	int n = 6;
//	gsl_vector **v = series_vectors(inp_dim, n, 1, 100, -10, 10);
//	print_series_vectors(v, n, "Vector series: ");

	gsl_vector *v = create_rand_vector(inp_dim, -10, 10);
	print_vector(v, "inp: ");

	input_vector_lstm(l, v);
	forward_pass_lstm(l);

	print_lstm(l);	

//	free_series_vectors(v, n);
	gsl_vector_free(v);
	free_lstm(l);

	return 0;
}
