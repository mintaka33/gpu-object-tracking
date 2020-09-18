
#include <string>
#include <fstream>
#include <iostream>
#include <complex>
#include <iomanip>

#include <stdio.h>
#include <math.h>

#include "perf.h"

using namespace std;

#define PI 3.14159265

PerfUtil pu;

struct Rect {
    int x;
    int y;
    int w;
    int h;
};

void dump2text(string filename, double* data, const int w, const int h) 
{
    char tmp[128] = {};
    ofstream of(filename);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            sprintf_s(tmp, "%14.4f", data[x + w * y]);
            of << tmp << ", ";
        }
        of << endl;
    }
    of.close();
}

void dump2yuv(char* dst, int w, int h, int i = 0)
{
    ofstream outfile;
    char filename[256] = {};
    sprintf_s(filename, "out_%d_%d_%02d.yuv", w, h, i);
    outfile.open(filename, ios::binary);
    outfile.write(dst, w*h);
    outfile.close();
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

void dft2d(const int M, const int N, double* f, double* F)
{
    for (size_t v = 0; v < N; v++) {
        for (size_t u = 0; u < M; u++) {
            complex<double> sum = 0;
            for (size_t y = 0; y < N; y++) {
                for (size_t x = 0; x < M; x++) {
                    double tmp = (u * x / (double)M + v * y / (double)N);
                    sum += f[y * M + x] * exp(complex<double>(0, -(2 * PI) * tmp));
                }
            }
            F[v*M + u] = sqrt(sum.real() * sum.real() + sum.imag() * sum.imag());
        }
    }
}

unsigned char bilinear(float q11, float q12, float q21, float q22, float x1, float y1, float x2, float y2, float x, float y)
{
    float r1, r2, p;
    r1 = (x2 - x)*q11 / (x2 - x1) + (x - x1)*q12 / (x2 - x1);
    r2 = (x2 - x)*q21 / (x2 - x1) + (x - x1)*q22 / (x2 - x1);
    p = (y2 - y)*r1 / (y2 - y1) + (y - y1)*r2 / (y2 - y1);

    if (p < 0) return 0;
    if (p > 255) return 255;
    return (unsigned char)p;
}

void affine(char* src, int sw, int sh, char* dst, int dw, int dh, float m[3][3])
{
    for (int j = 0; j < dh; j++) {
        for (int i = 0; i < dw; i++) {
            unsigned char yp = 0;
            float x1, y1, x2, y2, x, y;
            float q11[3], q12[3], q21[3], q22[3];
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

            q11[0] = (unsigned char)src[(y1i * sw + x1i)];
            q12[0] = (unsigned char)src[(y1i * sw + x2i)];
            q21[0] = (unsigned char)src[(y2i * sw + x1i)];
            q22[0] = (unsigned char)src[(y2i * sw + x2i)];
            yp = bilinear(q11[0], q12[0], q21[0], q22[0], x1, y1, x2, y2, x, y);

            dst[(j * dw + i)] = yp;
        }
    }
}

void random_affine()
{

}

void mosse_init(char* src, int srcw, int srch, Rect r)
{
    int cx = r.x + r.w / 2;
    int cy = r.y + r.h / 2;

    // Cosine window
    double* cw = new double[r.w * r.h];
    cos2d(cw, r.w, r.h);

    // Gaussian target
    double* g = new double[r.w * r.h];
    guassian2d(g, r.w, r.h);

    // DFT Gaussian target
    double* G = new double[r.w * r.h];
    memset(G, 0, sizeof(double) * r.w * r.h);
    //dft2d(r.w, r.h, g, G);

    double* Ai = new double[r.w * r.h];
    memset(Ai, 0, sizeof(double) * r.w * r.h);
    double* Bi = new double[r.w * r.h];
    memset(Bi, 0, sizeof(double) * r.w * r.h);

    // load original ROI
    char* roi = new char[r.w * r.h];
    for (size_t y = 0; y < r.h; y++) {
        for (size_t x = 0; x < r.w; x++) {
            roi[y * r.w + x] = src[(y+r.y)*srcw + (x+r.x)];
        }
    }
    dump2yuv(roi, r.w, r.h, 20);

    float angles[8] = { 0, -4.7, 3.8, -4.1, -0.9, 3.0, 0.5, -4.8 };
    for (size_t i = 0; i < 1; i++) {
        float d = angles[i] * (PI / 180);
        float m[3][3] = {
            { cos(d), sin(d), 0 },
            {-sin(d), cos(d), 0 },
            {      0,      0, 1 }
        };
        char* dst = new char[r.w * r.h];
        memset(dst, 0, r.w * r.h);
        affine(roi, r.w, r.h, dst, r.w, r.h, m);
        dump2yuv(dst, r.w, r.h, i);

    }

    delete[] roi;
}

void mosse_update()
{

}

void test_cos2d()
{
    const int w = 100, h = 100;
    double* cos = new double[h * w];

    for (size_t i = 0; i < 100; i++) {
        pu.startTick("cos2d");
        cos2d(cos, w, h);
        pu.stopTick("cos2d");
        printf("-");
    }

    pu.savePerfData();
    delete[] cos;
}

void test_guass()
{
    const int w = 100, h = 100;
    double* guass = new double[h * w];

    for (size_t i = 0; i < 100; i++) {
        pu.startTick("guass2d");
        guassian2d(guass, w, h);
        pu.stopTick("guass2d");
        printf("-");
    }

    dump2text("guass.csv", guass, w, h);
    pu.savePerfData();
    delete[] guass;
}

void test_dft()
{
    const int N = 64 * 64;
    double* x = new double[N];
    double* w = new double[N];
    for (size_t i = 0; i < N; i++) {
        x[i] = i % 256;
    }

    for (size_t i = 0; i < 1; i++)
    {
        pu.startTick("dft");
        dft(N, x, w);
        pu.stopTick("dft");
    }

    dump2text("dft.txt", w, 64, 64);

    delete[] x;
    delete[] w;
}

void test_dft2d()
{
    const int w = 30, h = 20;
    double* f = new double[w * h];
    double* F = new double[w * h];
    for (size_t i = 0; i < w*h; i++) {
        f[i] = i%256;
    }

    for (size_t i = 0; i < 10; i++)
    {
        pu.startTick("dft-2d");
        dft2d(w, h, f, F);
        pu.stopTick("dft-2d");
    }

    dump2text("dft2d.txt", F, w, h);

    delete[] f, F;
}

void test_affine()
{
    int srcw = 320, srch = 240;
    char* src = new char[srcw * srch];
    memset(src, 0, srcw * srch);
    char* dst = nullptr;
    int dstw = 0, dsth = 0;

    ifstream infile;
    infile.open("test.yuv", ios::binary);
    infile.read(src, srcw * srch);
    infile.close();

    // Translation
    int tx = 20;
    int ty = 10;
    float translation[3][3] = {
        { 1,  0,  tx},
        { 0,  1,  ty},
        { 0,  0,   1}
    };
    dstw = srcw;
    dsth = srch;
    dst = new char[dstw * dsth];
    affine(src, srcw, srch, dst, dstw, dsth, translation);
    dump2yuv(dst, dstw, dsth, 1);
    delete[] dst;

    // Scale
    float sx = 0.5;
    float sy = 1.5;
    float scale[3][3] = {
        {sx, 0,  0},
        { 0, sy, 0},
        { 0, 0,  1}
    };
    dstw = srcw;
    dsth = srch;
    dst = new char[dstw * dsth];
    memset(dst, 0, dstw * dsth);
    affine(src, srcw, srch, dst, dstw, dsth, scale);
    dump2yuv(dst, dstw, dsth, 2);
    delete[] dst;

    // Shear
    float shx = 0.2;
    float shy = 0.1;
    float shear[3][3] = {
        {  1, shx,  0 },
        {shy,   1,  0 },
        {  0,   0,  1 }
    };
    dstw = srcw;
    dsth = srch;
    dst = new char[dstw * dsth];
    affine(src, srcw, srch, dst, dstw, dsth, shear);
    dump2yuv(dst, dstw, dsth, 3);
    delete[] dst;

    // Rotation
    float d = -5 * (PI / 180);
    float rotation[3][3] = {
        { cos(d), sin(d), 0 },
        {-sin(d), cos(d), 0 },
        {      0,      0, 1 },
    };
    dstw = srcw;
    dsth = srch;
    dst = new char[dstw * dsth];
    affine(src, srcw, srch, dst, dstw, dsth, rotation);
    dump2yuv(dst, dstw, dsth, 4);
    delete[] dst;

    delete[] src;
}

void test_mosse()
{
    int srcw = 640, srch = 360;
    char* src = new char[srcw * srch];
    memset(src, 0, srcw * srch);

    ifstream infile;
    infile.open("tmp1.yuv", ios::binary);
    infile.read(src, srcw * srch);
    infile.close();

    //dump2yuv(src, srcw, srch, 1);

    Rect rect = { 196, 35, 145, 179 };
    mosse_init(src, srcw, srch, rect);

    delete[] src;
    return;
}

int main(int argc, int** argv) 
{
    test_mosse();

    printf("\ndone\n");
    return 0;
}
