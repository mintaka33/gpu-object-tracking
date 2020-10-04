#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <complex>
#include <iomanip>

#include <complex.h>

#include "perf.h"
#include "mosse.h"

#define PI 3.14159265

const int maxSize = 4096;

void dump2text(char* tag, double* data, const int w, const int h, int i = 0)
{
    char filename[256] = {};
    sprintf_s(filename, "dump.%s.%04d.%d.%d.txt", tag, i, w, h);
    std::ofstream of(filename);
    char tmp[64] = {};
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            sprintf_s(tmp, "%14.6f", data[x + w * y]);
            of << tmp << ", ";
        }
        of << std::endl;
    }
    of.close();
}

void dump2yuv(char* dst, int w, int h, int i = 0)
{
    std::ofstream outfile;
    char filename[256] = {};
    sprintf_s(filename, "dump.%04d.%d.%d.yuv", i, w, h);
    outfile.open(filename, std::ios::binary);
    outfile.write(dst, w * h);
    outfile.close();
}

template <class T>
T* allocArray(int size)
{
    T* p = new T[size];
    if (!p)
        return nullptr;

    memset(p, 0, sizeof(T) * size);
    return p;
}

inline void freeArray(void *p)
{
    if (!p) {
        delete[] p;
        p = nullptr;
    }
}

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

void guassian2d(double* guass, const int w, const int h, double sigma = 2.0)
{
    PFU_ENTER;

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

Mosse::Mosse(int w, int h):
    w(w), h(h)
{
    if (w<1 || h<1 || w>maxSize || h>maxSize)
        return;

    int sz = w * h;
    cos = allocArray<double>(sz);
    g = allocArray<double>(sz);

    int sz2 = sz * 2; // size of complex number array
    G = allocArray<double>(sz2);
    H = allocArray<double>(sz2);
    H1 = allocArray<double>(sz2);
    H2 = allocArray<double>(sz2);

    if (!cos || !g || !G || !H1 || !H2 || !H)
        return;

    cosWindow(cos, w, h);
    guassian2d(g, w, h);
    dft2d(w, h, g, G);

    initStatus = true;
}

Mosse::~Mosse()
{
    freeArray(curImg);
    freeArray(cos);
    freeArray(G);
    freeArray(H);
    freeArray(H1);
    freeArray(H2);
}

int Mosse::init(char* frame, const Rect r)
{
    return 0;
}

int Mosse::update(char* frame, Rect& out)
{
    return 0;
}

void Mosse::dump()
{
    dump2text("cos", cos, w, h, dumpIndex);
    dump2text("g", g, w, h, dumpIndex);
    dump2text("G", G, 2 * w, h, dumpIndex);
    dumpIndex++;
}
