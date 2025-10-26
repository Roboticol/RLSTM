// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"

int main() {
	init_utils();

	LSTM* l = create_rand_lstm(1,1,-1,1,-1,1);

	forget_gate_lstm(l);
	randomize_in_lstm(l, 1, 10);
	print_matrix(l->wf, "wf ");
	print_matrix(l->uf, "uf ");
	print_vector(l->x, "x ");
	print_vector(l->bf, "bf ");
	print_vector(l->f, "");

	free_lstm(l);
	return 0;
}
