#include <complex>
#include <iomanip>
#include <random>
#include <chrono>

#include <complex.h>
#include <time.h>

#include "math.h"
#include "perf.h"
#include "util.h"

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
    double hw = double(w) / 2.0;
    double hh = double(h) / 2.0;
    double c = 1 / (2 * PI * sigma * sigma);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            double ep = ((x - hw) * (x - hw) + (y - hh) * (y - hh)) / (sigma * sigma);
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

void idft2d(const int M, const int N, double* F, double* f)
{
    PFU_ENTER;

    for (size_t v = 0; v < N; v++) {
        for (size_t u = 0; u < M; u++) {
            std::complex<double> sum = 0;
            for (size_t y = 0; y < N; y++) {
                for (size_t x = 0; x < M; x++) {
                    double tmp = (u * x / (double)M + v * y / (double)N);
                    std::complex<double> t = exp(std::complex<double>(0, (2 * PI) * tmp));
                    double a = F[y * M * 2 + x * 2 + 0];
                    double b = F[y * M * 2 + x * 2 + 1];
                    double c = t.real();
                    double d = t.imag();

                    sum += std::complex<double>((a * c - b * d), (a * d + b * c));
                }
            }
            f[v * M + u] = sum.real() / (M * N);
        }
    }

    PFU_LEAVE;
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

            //if ((x < 0) || (y < 0) || (x > (dw - 2)) || (y > (dh - 2))) {
            //    continue;
            //}

            x = (x < 0) ? 0.0 : x;
            y = (y < 0) ? 0.0 : y;
            x = (x > (dw - 2)) ? (dw - 2) : x;
            y = (y > (dh - 2)) ? (dh - 2) : y;

            x1 = trunc(x); x1i = (int)x1;
            y1 = trunc(y); y1i = (int)y1;
            x2 = x1 + 1; x2i = (int)x2;
            y2 = y1 + 1; y2i = (int)y2;

            dst[(y1i * dw + x1i)] = src[j * dw + i];

            //q11[0] = src[(y1i * sw + x1i)];
            //q12[0] = src[(y1i * sw + x2i)];
            //q21[0] = src[(y2i * sw + x1i)];
            //q22[0] = src[(y2i * sw + x2i)];
            //yp = bilinear(q11[0], q12[0], q21[0], q22[0], x1, y1, x2, y2, x, y);

            //dst[(j * dw + i)] = yp;
        }
    }
}

#ifdef USE_OPENCV
void cvAffine(double* src, int sw, int sh, double* dst, int dw, int dh, double m[2][3])
{
    Mat src_mat(sh, sw, CV_64FC1, src);
    Mat dst_mat = Mat::zeros(sh, sw, src_mat.type());
    Mat affine_mat = Mat(2, 3, CV_64FC1);
    affine_mat.at<double>(0, 0) = m[0][0];
    affine_mat.at<double>(0, 1) = m[0][1];
    affine_mat.at<double>(0, 2) = m[0][2];
    affine_mat.at<double>(1, 0) = m[1][0];
    affine_mat.at<double>(1, 1) = m[1][1];
    affine_mat.at<double>(1, 2) = m[1][2];
    warpAffine(src_mat, dst_mat, affine_mat, src_mat.size(), INTER_LINEAR, BORDER_REFLECT);
    for (size_t y = 0; y < sh; y++) {
        for (size_t x = 0; x < sw; x++) {
            dst[y * sw + x] = dst_mat.at<double>(y, x);
        }
    }
}

void cvFFT2d(const size_t w, const size_t h, double* f, double* F)
{
    Mat src(Size(w, h), CV_64FC1), dst;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            src.at<double>(y, x) = f[y * w + x];
        }
    }

    PFU_ENTER;
    cv::dft(src, dst, DFT_COMPLEX_OUTPUT);
    PFU_LEAVE;

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < 2 * w; x++) {
            F[y * w * 2 + x + 0] = dst.at<double>(y, x);
        }
    }

    return;
}

void cvIFFT2d(const size_t w, const size_t h, double* F, double* f)
{
    Mat src(Size(w, h), CV_64FC2), dst;
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w*2; x++) {
            src.at<double>(y, x) = F[y * w * 2 + x + 0];
        }
    }

    PFU_ENTER;
    cv::idft(src, dst, DFT_REAL_OUTPUT);
    PFU_LEAVE;

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            f[y * w + x + 0] = dst.at<double>(y, x)/(w*h);
        }
    }

    return;
}
#endif

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
