#include <complex>
#include <iomanip>
#include <complex.h>

#include "math.h"
#include "perf.h"

void hanning(const int m, double* d)
{
    for (size_t i = 0; i < m; i++) {
        d[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
    }
}

void cosWindow(double* cos, const int w, const int h)
{
    PFU_ENTER;

    double* cos_w = new double[w];
    double* cos_h = new double[h];
    hanning(w, cos_w);
    hanning(h, cos_h);

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            cos[x + w * y] = sqrt(cos_h[y] * cos_w[x]);
        }
    }
    delete[] cos_w;
    delete[] cos_h;

    PFU_LEAVE;
}

void guassian2d(double* guass, const int w, const int h)
{
    PFU_ENTER;

    const double sigma = 2.0;
    double c = 1 / (2 * PI * sigma * sigma);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            double ep = ((x - w / 2) * (x - w / 2) + (y - h / 2) * (y - h / 2)) / (sigma * sigma);
            guass[x + y * w] = exp(-0.5 * ep);
        }
    }

    PFU_LEAVE;
}

void dft2d(const int M, const int N, double* f, double* F)
{
    PFU_ENTER;

    for (size_t v = 0; v < N; v++) {
        for (size_t u = 0; u < M; u++) {
            std::complex<double> sum = 0;
            for (size_t y = 0; y < N; y++) {
                for (size_t x = 0; x < M; x++) {
                    double tmp = (u * x / (double)M + v * y / (double)N);
                    sum += f[y * M + x] * exp(std::complex<double>(0, -(2 * PI) * tmp));
                }
            }
            F[v * M * 2 + 2 * u + 0] = sum.real();
            F[v * M * 2 + 2 * u + 1] = sum.imag();
        }
    }

    PFU_LEAVE;
}
