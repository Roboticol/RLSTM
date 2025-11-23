// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"
#include "backprop.h"

int main() {
	init_utils();

	int inp_dim = 2;
	int hidden_dim = 2;
	int output_dim = 2;
	LSTM* l = create_rand_lstm(inp_dim, hidden_dim, output_dim,-1,1,-10,10);
	randomize_in_lstm(l, -1,1);


	gsl_vector *v = create_rand_vector(output_dim, -10, 10);
	gsl_vector *h = create_rand_vector(hidden_dim, -10, 10);
	print_vector(v, "inp: ");

	input_vector_lstm(l, v);

	forward_pass_lstm(l);
	bp_dEdh(l, v, h);

	print_lstm(l);	
	print_vector(h, "dE/dh: ");

	gsl_vector_free(v);
	free_lstm(l);

	return 0;
}
