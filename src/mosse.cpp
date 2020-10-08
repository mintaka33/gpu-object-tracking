#pragma once
#include <string>
#include <fstream>
#include <iostream>

#include "perf.h"
#include "mosse.h"
#include "math.h"

const int maxSize = 4096;

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
#if !USE_OPENCV
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
#endif
}

int Mosse::init(char* frame, int pw, int ph, const RoiRect r)
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
#if !USE_OPENCV
    cos = allocArray<double>(sz);
    g = allocArray<double>(sz);
    f = allocArray<double>(sz);
    fa = allocArray<double>(sz);
    fi = allocArray<double>(sz);
    gi = allocArray<double>(sz);
#else
    cosMat = new Mat(h, w, CV_64FC1);
    gMat = new Mat(h, w, CV_64FC1);
    fMat = new Mat(h, w, CV_64FC1);
    faMat = new Mat(h, w, CV_64FC1);
    fiMat = new Mat(h, w, CV_64FC1);
    giMat = new Mat(h, w, CV_64FC1);
    cos = (double*)cosMat->data;// allocArray<double>(sz);
    g = (double*)gMat->data;
    f = (double*)fMat->data;
    fa = (double*)faMat->data;
    fi = (double*)fiMat->data;
    gi = (double*)giMat->data;
    memset(gi, 0, sizeof(double) * w * h);
#endif

    int sz2 = sz * 2; // size of complex number array
#if !USE_OPENCV
    G = allocArray<double>(sz2);
    H = allocArray<double>(sz2);
    H1 = allocArray<double>(sz2);
    H2 = allocArray<double>(sz2);
    Fi = allocArray<double>(sz2);
    Gi = allocArray<double>(sz2);
#else
    GMat = new Mat(h, 2 * w, CV_64FC1);
    HMat = new Mat(h, 2 * w, CV_64FC1);
    H1Mat = new Mat(h, 2 * w, CV_64FC1);
    H2Mat = new Mat(h, 2 * w, CV_64FC1);
    FiMat = new Mat(h, 2 * w, CV_64FC1);
    GiMat = new Mat(h, 2 * w, CV_64FC1);
    G = (double*)GMat->data;
    H = (double*)HMat->data;
    H1 = (double*)H1Mat->data;
    H2 = (double*)H2Mat->data;
    Fi = (double*)FiMat->data;
    Gi = (double*)GiMat->data;
    memset(H1, 0, sizeof(double) * w * 2 * h);
    memset(H2, 0, sizeof(double) * w * 2 * h);
    memset(Gi, 0, sizeof(double) * w * 2 * h);
#endif

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
        double m[2][3] = {}; // { 1.027946, 0.003986, -0.542760, -0.142644, 1.008884, 1.864269 };
        getMatrix(w, h, m[0]);

        memset(fa, 0, sizeof(double) * w * h);
#if !USE_OPENCV
        affine(f, w, h, fa, w, h, m);
#else
        cvAffine(f, w, h, fa, w, h, m);
#endif

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
        for (size_t i = 0; i < w; i++) {
            double a =  H[j * 2 * w + 2 * i + 0];
            double b =  H[j * 2 * w + 2 * i + 1];
            double c = Fi[j * 2 * w + 2 * i + 0];
            double d = Fi[j * 2 * w + 2 * i + 1];
            Gi[j * 2 * w + 2 * i + 0] = a * c - b * d;
            Gi[j * 2 * w + 2 * i + 1] = a * d + b * c;
        }
    }

    // gi = IDFT(Gi)
    idft2d(w, h, Gi, gi);

    int mx = 0, my = 0;
    double max = gi[0];
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            if (gi[j * w + i] > max) {
                max = gi[j * w + i];
                mx = i;
                my = j;
            }
        }
    }

    printf("INFO: mx = %d, my = %d, dx = %d, dy = %d\n", mx, my, (mx - w/2), (my - h/2));

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
    dump2text("gi", gi, w, h, dumpIndex);

    dump2text("G",  G,  2 * w, h, dumpIndex);
    dump2text("Fi", Fi, 2 * w, h, dumpIndex);
    dump2text("H1", H1, 2 * w, h, dumpIndex);
    dump2text("H2", H2, 2 * w, h, dumpIndex);
    dump2text("H",  H,  2 * w, h, dumpIndex);
    dump2text("Gi", Gi, 2 * w, h, dumpIndex);

    dumpIndex++;
}

