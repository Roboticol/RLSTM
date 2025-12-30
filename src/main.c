// Made by Roboticol
// make sure to check out the backprop.h/c, nutils.h/c and lstm.h/c files!

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"
#include "backprop.h"

int main() {
	init_utils(); //initialize the utilities library

	// dimensions
	int input_dim = 1; // takes just one input
	int output_dim = 1; // outputs one value
	int hidden_dim = 3; // hidden state is a vector of 3 elements
	
	// initialize an lstm with randomized weights matrices and biases vectors
	LSTM *lstm = create_rand_lstm(input_dim, hidden_dim, output_dim, -10, 10, -10, 10);
	// display lstm values
	print_lstm(lstm);

	// create data for the lstm to train on
	gsl_vector **series = series_vectors(1, 10, 1, 10, 1, 10); // generates a series of 10 vectors of size 1 and all its values are randomized between 1 and 10
	// display series data
	print_series_vectors(series, 10, "\n\nData: ");

	// have the lstm predict its result on the series
	forward_pass_n_lstm(lstm, series, 10);
	
	// print lstm after forward pass
	print_lstm(lstm);

	// output final value of lstm
	print_vector(lstm->y, "Output of lstm: ");

	// memory management steps
	free_lstm(lstm);
	free_series_vectors(series, 10);

	return 0;
}
