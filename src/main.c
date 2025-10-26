// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"

int main() {
	init_utils();

	LSTM* l = create_rand_lstm(2,5,-1,1,-1,1);

	int n = 10;
	gsl_vector **v = series_vectors(3, n, 1, 100, -10, 10);
	print_series_vectors(v, n, "series: ");
	free_series_vectors(v, n);
	free_lstm(l);

	return 0;
}
