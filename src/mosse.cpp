#pragma once
#include <string>
#include <fstream>
#include <iostream>

#include "perf.h"
#include "mosse.h"
#include "math.h"
#include "util.h"

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

void cropNorm(char* src, int picW, int picH, RoiRect r, double* dst)
{
    for (size_t j = 0; j < r.h; j++) {
        for (size_t i = 0; i < r.w; i++) {
            dst[j * r.w + i] = (double((uint8_t)src[(j + r.y) * picW + (i + r.x)])) / 255.0;
        }
    }
}

Mosse::Mosse()
{

}

Mosse::~Mosse()
{
#ifdef USE_OPENCV

#else
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
    int sz2 = sz * 2; // size of complex number array

#ifdef USE_OPENCV
    imgMat = new Mat(picH, picW, CV_8UC1);
    cosMat = new Mat(h, w, CV_64FC1);
    gMat = new Mat(h, w, CV_64FC1);
    fMat = new Mat(h, w, CV_64FC1);
    faMat = new Mat(h, w, CV_64FC1);
    fiMat = new Mat(h, w, CV_64FC1);
    giMat = new Mat(h, w, CV_64FC1);

    curImg = (char*)imgMat->data;
    cos = (double*)cosMat->data;
    g = (double*)gMat->data;
    f = (double*)fMat->data;
    fa = (double*)faMat->data;
    fi = (double*)fiMat->data;
    gi = (double*)giMat->data;

    memset(gi, 0, sizeof(double) * w * h);

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
#else
    curImg = allocArray<char>(sz);
    cos = allocArray<double>(sz);
    g = allocArray<double>(sz);
    f = allocArray<double>(sz);
    fa = allocArray<double>(sz);
    fi = allocArray<double>(sz);
    gi = allocArray<double>(sz);

    G = allocArray<double>(sz2);
    H = allocArray<double>(sz2);
    H1 = allocArray<double>(sz2);
    H2 = allocArray<double>(sz2);
    Fi = allocArray<double>(sz2);
    Gi = allocArray<double>(sz2);
#endif

    if (!cos || !g || !f || !fa || !fi || !G || !H || !H1 || !H2 || !Fi || !Gi)
        return -1;

    memcpy_s(curImg, picW*picH, frame, picW * picH);

    cosWindow(cos, w, h);
    guassian2d(g, w, h);

#ifdef USE_OPENCV
    cvFFT2d(w, h, g, G);
#else
    dft2d(w, h, g, G);
#endif

    // load ROI and normalization
    cropNorm(frame, picW, picH, r, f);

    for (size_t i = 0; i < affineNum; i++) {
        double m[2][3] = {}; // { 1.027946, 0.003986, -0.542760, -0.142644, 1.008884, 1.864269 };
        getMatrix(w, h, m[0]);

        memset(fa, 0, sizeof(double) * w * h);
#ifdef USE_OPENCV
        cvAffine(f, w, h, fa, w, h, m);
#else
        affine(f, w, h, fa, w, h, m);
#endif

        preproc(fa, cos, fi, w, h);
#ifdef USE_OPENCV
        cvFFT2d(w, h, fi, Fi);
#else
        dft2d(w, h, fi, Fi);
#endif

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

    initStatus = true;
    frameIndex++;

    return 0;
}

int Mosse::update(char* frame, int pw, int ph, RoiRect& out)
{
    if (pw != picW || ph != picH)
        return -1;

    memcpy_s(curImg, pw * ph, frame, pw * ph);

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

    // Fi
    RoiRect rt = { x, y, w, h };
    cropNorm(frame, picW, picH, rt, f);
    preproc(f, cos, fi, w, h);

#ifdef USE_OPENCV
    cvFFT2d(w, h, fi, Fi);
#else
    dft2d(w, h, fi, Fi);
#endif

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
#ifdef USE_OPENCV
    cvIFFT2d(w, h, Gi, gi);
#else
    idft2d(w, h, Gi, gi);
#endif

    // find peak value position
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
    int dx = (int)round((double)mx - ((double)w) / 2);
    int dy = (int)round((double)my - ((double)h) / 2);
    printf("INFO: mx = %d, my = %d, dx = %d, dy = %d\n", mx, my, dx, dy);

    x = x + dx;
    y = y + dy;
    out = { x, y, w, h };

    cropNorm(frame, picW, picH, out, f);
    preproc(f, cos, fi, w, h);

#ifdef USE_OPENCV
    cvFFT2d(w, h, fi, Fi);
#else
    dft2d(w, h, fi, Fi);
#endif

    // update H1, H2
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            // H1 = rate * (G  * np.conj(Fi)) + (1 - rate) * H1
            // H2 = rate * (Fi * np.conj(Fi)) + (1 - rate) * H2
            double a =  G[j * w * 2 + i * 2 + 0];
            double b =  G[j * w * 2 + i * 2 + 1];
            double c = Fi[j * w * 2 + i * 2 + 0];
            double d = Fi[j * w * 2 + i * 2 + 1];

            // (a+bi)*(c-di) = (ac + bd) + (bc-ad)i
            double gfr = a * c + b * d;
            double gfi = b * c - a * d;

            // (c+di)*(c-di) = (cc+dd)i
            double ffr = c * c + d * d;
            double ffi = 0;

            double h1r = H1[j * w * 2 + i * 2 + 0];
            double h1i = H1[j * w * 2 + i * 2 + 1];
            double h2r = H2[j * w * 2 + i * 2 + 0];
            double h2i = H2[j * w * 2 + i * 2 + 1];

            H1[j * w * 2 + i * 2 + 0] = rate * gfr + (1 - rate) * h1r;
            H1[j * w * 2 + i * 2 + 1] = rate * gfi + (1 - rate) * h1i;
            H2[j * w * 2 + i * 2 + 0] = rate * ffr + (1 - rate) * h2r;
            H2[j * w * 2 + i * 2 + 1] = rate * ffi + (1 - rate) * h2i;
        }
    }

    frameIndex++;
    return 0;
}

void Mosse::dumpResult()
{
#ifdef USE_OPENCV
    char filename[256] = {};
    sprintf_s(filename, "output/tmp.out.%03d.bmp", frameIndex);
    cv::rectangle(*imgMat, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0));
    cv::imwrite(filename, *imgMat);

#endif
}

void Mosse::dump2bin()
{
    uint8_t* t = new uint8_t[w * h];

    double2uchar(t, f, w, h);
    dump2yuv("f", t, w, h, frameIndex);

    double2uchar(t, fa, w, h);
    dump2yuv("fa", t, w, h, frameIndex);

    delete[] t;
}

void Mosse::dump2txt()
{
    dump2text("cos", cos, w, h, frameIndex);
    dump2text("g", g, w, h, frameIndex);
    dump2text("f", f, w, h, frameIndex);
    dump2text("fa", fa, w, h, frameIndex);
    dump2text("fi", fi, w, h, frameIndex);
    dump2text("gi", gi, w, h, frameIndex);

    dump2text("G",  G,  2 * w, h, frameIndex);
    dump2text("Fi", Fi, 2 * w, h, frameIndex);
    dump2text("H1", H1, 2 * w, h, frameIndex);
    dump2text("H2", H2, 2 * w, h, frameIndex);
    dump2text("H",  H,  2 * w, h, frameIndex);
    dump2text("Gi", Gi, 2 * w, h, frameIndex);
}

