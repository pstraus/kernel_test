#pragma once

#include <iostream>
#include <math.h>

namespace kernel_test
{
// Kernel function to add the elements of two arrays
__global__ 
void add(int N, float* x, float* y, float* out);
}

