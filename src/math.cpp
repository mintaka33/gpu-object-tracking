
// https://numpy.org/doc/stable/reference/generated/numpy.hanning.html

#include <iostream>
#include <stdio.h>
#include <math.h>

#include "perf.h"

#define PI 3.14159265

PerfUtil pu;

void print(double* data, const int w, const int h) {
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            printf("%f, ", data[x + w * y]);
        }
        printf("\n");
    }
}

void hanning(const int m, double* d)  {
    for (size_t i = 0; i < m; i++) {
        d[i] = 0.5 - 0.5 * cos(2*PI*i/(m-1));
    }
}

void cos2d(double* cos, const int w, const int h) {
    double* cos_w = new double[w];
    double* cos_h = new double[h];
    hanning(w, cos_w);
    hanning(h, cos_h);

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            cos[x + w*y] = cos_h[y] * cos_w[x];
        }
    }
    delete[] cos_w;
    delete[] cos_h;
}

void guassian2d(double* guass, const int w, const int h) {
    const double sigma = 2.0;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            double ep = ((x-w/2) * (x-w/2) + (y-h/2)*(y-h/2))/(sigma * sigma);
            guass[x + y * w] = exp(ep);
        }
    }
}

int main(int argc, int** argv) {
    const int w = 300, h = 200;
    double* cos = new double[h*w];
    double* guass = new double[h * w];

    for (size_t i = 0; i < 1000; i++) {

        pu.startTick("cos2d");
        cos2d(cos, w, h);
        pu.stopTick("cos2d");

        pu.startTick("guass2d");
        guassian2d(guass, w, h);
        pu.stopTick("guass2d");

        printf("-");
    }

    pu.savePerfData();

    delete[] cos;
    delete[] guass;

    printf("\ndone\n");
    return 0;
}
