// Made by Roboticol

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include "lstm.h"
#include "nutils.h"

int main() {
	int data[] = {5,6,9,1,2,6,4,3,2};
	testfunc();		

	gsl_vector *a = gsl_vector_calloc(3);
	gsl_vector *b = gsl_vector_calloc(2);

	gsl_vector_set(a, 0, 2);
	gsl_vector_set(a, 1, 1);
	gsl_vector_set(a, 2, 0.23);

	gsl_vector_set(b, 0, 2.58);
	gsl_vector_set(b, 1, 3.23);

	gsl_vector *c = concatenate_vector(a, b);
	print_vector(c);
	sigmoid_vector(c, c);
	print_vector(c);

	printf("\n");

	gsl_vector_free(c);
	gsl_vector_free(b);
	gsl_vector_free(a);
	
	return 0;
}
