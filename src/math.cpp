
#include <string>
#include <fstream>
#include <iostream>
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

int main(int argc, int** argv) {
    const int w = 100, h = 100;
    double* cos = new double[h*w];
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

    printf("\ndone\n");
    return 0;
}
