#include <complex>
#include <iomanip>
#include <complex.h>
#include <time.h>

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

void genMatrix(int w, int h, double* mat)
{
    double r[5] = {};
    srand(time(NULL));
    for (size_t i = 0; i < 5; i++) {
        r[i] = (((double)rand() / (RAND_MAX)) - 0.5) * 0.2; // (-0.1, 0.1)
    }

    double c = cos(r[0]);
    double s = sin(r[0]);
    double m[2][3] = {};
    m[0][0] = c + r[1];
    m[0][1] = -s + r[2];
    m[1][0] = s + r[3];
    m[1][1] = c + r[4];
    double c1 = w / 2.0, c2 = h / 2.0;
    double t1 = m[0][0] * c1 + m[0][1] * c2;
    double t2 = m[1][0] * c1 + m[1][1] * c2;
    m[0][2] = c1 - t1;
    m[1][2] = c2 - t2;

    mat[0] = m[0][0], mat[1] = m[0][1], mat[2] = m[0][2];
    mat[3] = m[1][0], mat[4] = m[1][1], mat[5] = m[1][2];
}

double bilinear(double q11, double q12, double q21, double q22, double x1, double y1, double x2, double y2, double x, double y)
{
    double r1, r2, p;
    r1 = (x2 - x) * q11 / (x2 - x1) + (x - x1) * q12 / (x2 - x1);
    r2 = (x2 - x) * q21 / (x2 - x1) + (x - x1) * q22 / (x2 - x1);
    p = (y2 - y) * r1 / (y2 - y1) + (y - y1) * r2 / (y2 - y1);
    return p;
}

void affine(double* src, int sw, int sh, double* dst, int dw, int dh, double m[2][3])
{
    for (int j = 0; j < dh; j++) {
        for (int i = 0; i < dw; i++) {
            double yp = 0;
            double x1, y1, x2, y2, x, y;
            double q11[3], q12[3], q21[3], q22[3];
            int x1i, y1i, x2i, y2i;

            x = m[0][0] * i + m[0][1] * j + m[0][2] * 1;
            y = m[1][0] * i + m[1][1] * j + m[1][2] * 1;

            if ((x < 0) || (y < 0) || (x > (dw - 2)) || (y > (dh - 2))) {
                continue;
            }

            x = (x < 0) ? 0.0 : x;
            y = (y < 0) ? 0.0 : y;
            x = (x > (dw - 2)) ? (dw - 2) : x;
            y = (y > (dh - 2)) ? (dh - 2) : y;

            x1 = trunc(x); x1i = (int)x1;
            y1 = trunc(y); y1i = (int)y1;
            x2 = x1 + 1; x2i = (int)x2;
            y2 = y1 + 1; y2i = (int)y2;

            q11[0] = src[(y1i * sw + x1i)];
            q12[0] = src[(y1i * sw + x2i)];
            q21[0] = src[(y2i * sw + x1i)];
            q22[0] = src[(y2i * sw + x2i)];
            yp = bilinear(q11[0], q12[0], q21[0], q22[0], x1, y1, x2, y2, x, y);

            dst[(j * dw + i)] = yp;
        }
    }
}

void preproc(double* f, double* cos, double* dst, int w, int h)
{
    const double eps = 1e-5;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            dst[y * w + x] = log(double(f[y * w + x]) + 1);
        }
    }

    double avg = 0;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            avg += dst[y * w + x];
        }
    }
    avg = avg / (w * h);

    double sd = 0;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            sd += (dst[y * w + x] - avg) * (dst[y * w + x] - avg);
        }
    }
    sd = sqrt(sd / (w * h));

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            dst[y * w + x] = ((dst[y * w + x] - avg) / (sd + eps)) * cos[y * w + x];
        }
    }
}
