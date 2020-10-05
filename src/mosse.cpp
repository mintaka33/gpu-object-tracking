#pragma once
#include <string>
#include <fstream>
#include <iostream>

#include "perf.h"
#include "mosse.h"
#include "math.h"

const int maxSize = 4096;

void dump2text(char* tag, double* data, const int w, const int h, int i = 0)
{
    char filename[256] = {};
    sprintf_s(filename, "dump.%s.%04d.%dx%d.txt", tag, i, w, h);
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

void dump2yuv(char* tag, uint8_t* dst, int w, int h, int i = 0)
{
    std::ofstream outfile;
    char filename[256] = {};
    sprintf_s(filename, "dump.%s.%04d.%dx%d.yuv", tag, i, w, h);
    outfile.open(filename, std::ios::binary);
    outfile.write((char*)dst, w * h);
    outfile.close();
}

void double2uchar(uint8_t *dst, double *src, int w, int h)
{
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            double a = src[j * w + i] * 255;
            dst[j * w + i] = (a > 255) ? 255 : ((a < 0) ? 0 : uint8_t(a));
        }

    }
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

Mosse::Mosse()
{

}

Mosse::~Mosse()
{
    freeArray(curImg);
    freeArray(cos);
    freeArray(f);
    freeArray(fa);
    freeArray(fi);
    freeArray(Fi);
    freeArray(G);
    freeArray(Gi);
    freeArray(H);
    freeArray(H1);
    freeArray(H2);
}

int Mosse::init(char* frame, int pw, int ph, const Rect r)
{
    picW = pw;
    picH = ph;
    x = r.x;
    y = r.y;
    w = r.w;
    h = r.h;

    if (w<1 || h<1 || w>maxSize || h>maxSize)
        return -1;

    int sz = w * h;
    cos = allocArray<double>(sz);
    g = allocArray<double>(sz);
    f = allocArray<double>(sz);
    fa = allocArray<double>(sz);
    fi = allocArray<double>(sz);

    int sz2 = sz * 2; // size of complex number array
    G = allocArray<double>(sz2);
    H = allocArray<double>(sz2);
    H1 = allocArray<double>(sz2);
    H2 = allocArray<double>(sz2);
    Fi = allocArray<double>(sz2);
    Gi = allocArray<double>(sz2);

    if (!cos || !g || !f || !fa || !fi || !G || !H || !H1 || !H2 || !Fi || !Gi)
        return -1;

    cosWindow(cos, w, h);
    guassian2d(g, w, h);
    dft2d(w, h, g, G);

    // load ROI and normalization
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            f[j * w + i] = (double((uint8_t)frame[(j + y) * picW + (i + x)])) / 255.0;
        }
    }

    for (size_t i = 0; i < affineNum; i++) {
        double m[2][3] = { 1.027946, 0.003986, -0.542760, -0.142644, 1.008884, 1.864269 };
        getMatrix(w, h, m[0]);

        memset(fa, 0, sizeof(double) * w * h);
        affine(f, w, h, fa, w, h, m);

        preproc(fa, cos, fi, w, h);

        dft2d(w, h, fi, Fi);

        // calculate H1, H2
        for (size_t j = 0; j < h; j++) {
            for (size_t i = 0; i < w; i++) {
                // H1 += G * np.conj(Fi)
                // H2 += Fi * np.conj(Fi)
                double a =  G[j * w * 2 + i * 2 + 0];
                double b =  G[j * w * 2 + i * 2 + 1];
                double c = Fi[j * w * 2 + i * 2 + 0];
                double d = Fi[j * w * 2 + i * 2 + 1];
                // (a+bi)*(c-di) = (ac + bd) + (bc-ad)i
                H1[j * w * 2 + i * 2 + 0] += a * c + b * d;
                H1[j * w * 2 + i * 2 + 1] += b * c - a * d;
                // (c+di)*(c-di) = (cc+dd)i
                H2[j * w * 2 + i * 2 + 0] += c * c + d * d;
                H2[j * w * 2 + i * 2 + 1] += 0;
            }
        }
    }

    // calculate H = H1 / H2
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            double a = H1[j * 2 * w + 2 * i + 0];
            double b = H1[j * 2 * w + 2 * i + 1];
            double c = H2[j * 2 * w + 2 * i + 0];
            double d = H2[j * 2 * w + 2 * i + 1];
            H[j * 2 * w + 2 * i + 0] = (a * c + b * d) / (c * c + d * d);
            H[j * 2 * w + 2 * i + 1] = (b * c - a * d) / (c * c + d * d);
        }
    }

    initStatus = true;

    return 0;
}

int Mosse::update(char* frame, int pw, int ph)
{
    if (pw != picW || ph != picH)
        return -1;

    // Fi
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            f[j * w + i] = (double((uint8_t)frame[(j + y) * picW + (i + x)])) / 255.0;
        }
    }
    preproc(f, cos, fi, w, h);
    dft2d(w, h, fi, Fi);

    // Gi = H * Fi
    for (size_t j = 0; j < h; j++) {
        for (size_t x = 0; x < w; x++) {
            double a =  H[j * 2 * w + 2 * x + 0];
            double b =  H[j * 2 * w + 2 * x + 1];
            double c = Fi[j * 2 * w + 2 * x + 0];
            double d = Fi[j * 2 * w + 2 * x + 1];
            Gi[j * 2 * w + 2 * x + 0] = a * c - b * d;
            Gi[j * 2 * w + 2 * x + 1] = a * d + b * c;
        }
    }

    // gi = IDFT(Gi)
    double* gi = allocArray<double>(w * h);
    idft2d(w, h, Gi, gi);

    int mx = 0, my = 0, max = gi[0];
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            if (gi[j * w + i] > max) {
                max = gi[j * w + i];
                mx = i;
                my = j;
            }
        }
    }

    printf("INFO: mx = %d, my = %d\n", mx, my);

    freeArray(gi);
    return 0;
}

void Mosse::dump2bin()
{
    uint8_t* t = new uint8_t[w * h];

    double2uchar(t, f, w, h);
    dump2yuv("f", t, w, h, dumpIndex);

    double2uchar(t, fa, w, h);
    dump2yuv("fa", t, w, h, dumpIndex);

    delete[] t;
}

void Mosse::dump2txt()
{
    dump2text("cos", cos, w, h, dumpIndex);
    dump2text("g", g, w, h, dumpIndex);
    dump2text("f", f, w, h, dumpIndex);
    dump2text("fa", fa, w, h, dumpIndex);
    dump2text("fi", fi, w, h, dumpIndex);

    dump2text("G",  G,  2 * w, h, dumpIndex);
    dump2text("Fi", Fi, 2 * w, h, dumpIndex);
    dump2text("H1", H1, 2 * w, h, dumpIndex);
    dump2text("H2", H2, 2 * w, h, dumpIndex);
    dump2text("H",  H,  2 * w, h, dumpIndex);

    dumpIndex++;
}

