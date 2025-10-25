// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"

int main() {
	init_utils();
	int data[] = {5,6,9,1,2,6,4,3,2};
	
	gsl_vector *v = gsl_vector_calloc(5);

	randomize_vector(v, 1, 5);
	print_vector(v);
	randomize_vector(v, 1, 5);
	print_vector(v);

	printf("\n");
	gsl_matrix *m = create_rand_matrix(5,2,1,100);
	print_matrix(m);

	gsl_matrix_free(m);
	gsl_vector_free(v);
	return 0;
}
