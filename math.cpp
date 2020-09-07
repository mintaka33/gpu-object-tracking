// math.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// https://numpy.org/doc/stable/reference/generated/numpy.hanning.html

#include <iostream>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265

#define W 300
#define H 200

void hanning(int m, double* d)
{
    for (size_t i = 0; i < m; i++)
    {
        d[i] = 0.5 - 0.5 * cos(2*PI*i/(m-1));
    }
}

void cos2d(int w, int h, double* d)
{
    double cos_w[W] = {};
    double cos_h[H] = {};
    hanning(W, cos_w);
    hanning(H, cos_h);

    for (size_t y = 0; y < H; y++)
    {
        for (size_t x = 0; x < W; x++)
        {
            d[x + W*y] = cos_h[y] * cos_w[x];
        }
    }
}

void print(double* cos)
{
    for (size_t y = 0; y < H; y++)
    {
        for (size_t x = 0; x < W; x++)
        {
            printf("%f, ", cos[x + W * y]);
        }
        printf("\n");
    }
}

int main()
{
    double* cos = new double[H*W];
    cos2d(W, H, cos);

    printf("done\n");
    delete [] cos;
    return 0;
}
