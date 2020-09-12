
#include <string>
#include <fstream>
#include <iostream>
#include <complex>

#include <stdio.h>
#include <math.h>

#include "perf.h"

using namespace std;

#define PI 3.14159265

PerfUtil pu;

void dump2file(string filename, double* data, const int w, const int h) {
    ofstream of(filename);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            of << data[x + w * y] << ", ";
        }
        of << endl;
    }
    of.close();
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
    double c = 1 / (2 * PI * sigma * sigma);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            double ep = ((x-w/2) * (x-w/2) + (y-h/2)*(y-h/2))/(sigma * sigma);
            guass[x + y * w] = c * exp(-0.5 * ep);
        }
    }
}

void dft(const int N, double* x, double* abs)
{
    for (int k = 0; k < N; k++) {
        complex<double> sum = 0;
        for (int n = 0; n < N; n++) {
            sum += x[n] * exp(complex<double>(0, -(2 * PI / N) * n * k));
        }
        abs[k] = sqrt( sum.real() * sum.real() + sum.imag() * sum.imag());
    }
}

void test()
{
    const int w = 100, h = 100;
    double* cos = new double[h * w];
    double* guass = new double[h * w];

    for (size_t i = 0; i < 100; i++) {

        pu.startTick("cos2d");
        cos2d(cos, w, h);
        pu.stopTick("cos2d");

        pu.startTick("guass2d");
        guassian2d(guass, w, h);
        pu.stopTick("guass2d");

        printf("-");
    }

    dump2file("guass.csv", guass, w, h);

    pu.savePerfData();

    delete[] cos;
    delete[] guass;
}

int main(int argc, int** argv) {
    const int N = 1024;
    double x[N] = {}, w[N] = {};
    for (size_t i = 0; i < N; i++){
        x[i] = i;
    }

    for (size_t i = 0; i < 10; i++)
    {
        pu.startTick("dft");
        dft(N, x, w);
        pu.stopTick("dft");
    }


    printf("\ndone\n");
    return 0;
}
