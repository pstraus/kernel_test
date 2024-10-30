#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y, float *out);
