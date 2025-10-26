// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"

int main() {
	init_utils();

	LSTM* l = create_rand_lstm(2,4,1,100,1,100);
	printf("sfae: %d\n", l->wf->size1);
	print_lstm(l);
	free_lstm(l);
	return 0;
}
