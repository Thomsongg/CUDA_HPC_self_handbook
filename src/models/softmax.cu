#include <stdio.h>
#include <time.h>
#include <iostream>

#define CEIL(a, b) ((a + b - 1) / b)
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void softmax(float *input, float *output, float *max_val, int N)
{

}

int main()
{
    return 0;
}