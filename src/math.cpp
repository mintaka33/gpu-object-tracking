
#include <string>
#include <fstream>
#include <iostream>
#include <complex>
#include <iomanip>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

struct Buf2D {
    char* buf;
    int w;
    int h;
};

void dump2text(string filename, float* data, const int w, const int h) 
{
    char tmp[128] = {};
    ofstream of(filename);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            sprintf_s(tmp, "%14.6f", data[x + w * y]);
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

void hanning(const int m, float* d)  {
    for (size_t i = 0; i < m; i++) {
        d[i] = 0.5 - 0.5 * cos(2*PI*i/(m-1));
    }
}

void cos2d(float* cos, const int w, const int h) {
    float* cos_w = new float[w];
    float* cos_h = new float[h];
    hanning(w, cos_w);
    hanning(h, cos_h);

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            cos[x + w*y] = sqrt(cos_h[y] * cos_w[x]);
        }
    }
    delete[] cos_w;
    delete[] cos_h;
}

void guassian2d(float* guass, const int w, const int h) {
    const float sigma = 2.0;
    float c = 1 / (2 * PI * sigma * sigma);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            float ep = ((x-w/2) * (x-w/2) + (y-h/2)*(y-h/2))/(sigma * sigma);
            guass[x + y * w] = exp(-0.5 * ep);
        }
    }
}

void dft(const int N, float* x, float* abs)
{
    for (int k = 0; k < N; k++) {
        complex<float> sum = 0;
        for (int n = 0; n < N; n++) {
            sum += x[n] * exp(complex<float>(0, -(2 * PI / N) * n * k));
        }
        abs[k] = sqrt( sum.real() * sum.real() + sum.imag() * sum.imag());
    }
}

void dft2d(const int M, const int N, float* f, float* F)
{
    for (size_t v = 0; v < N; v++) {
        for (size_t u = 0; u < M; u++) {
            complex<float> sum = 0;
            for (size_t y = 0; y < N; y++) {
                for (size_t x = 0; x < M; x++) {
                    float tmp = (u * x / (float)M + v * y / (float)N);
                    sum += f[y * M + x] * exp(complex<float>(0, -(2 * PI) * tmp));
                }
            }
            F[v * M * 2 + 2 * u + 0] = sum.real();
            F[v * M * 2 + 2 * u + 1] = sum.imag();
        }
    }
}

float bilinear(float q11, float q12, float q21, float q22, float x1, float y1, float x2, float y2, float x, float y)
{
    float r1, r2, p;
    r1 = (x2 - x)*q11 / (x2 - x1) + (x - x1)*q12 / (x2 - x1);
    r2 = (x2 - x)*q21 / (x2 - x1) + (x - x1)*q22 / (x2 - x1);
    p = (y2 - y)*r1 / (y2 - y1) + (y - y1)*r2 / (y2 - y1);
    return p;
}

void affine(float* src, int sw, int sh, float* dst, int dw, int dh, float m[3][3])
{
    for (int j = 0; j < dh; j++) {
        for (int i = 0; i < dw; i++) {
            float yp = 0;
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

            q11[0] = src[(y1i * sw + x1i)];
            q12[0] = src[(y1i * sw + x2i)];
            q21[0] = src[(y2i * sw + x1i)];
            q22[0] = src[(y2i * sw + x2i)];
            yp = bilinear(q11[0], q12[0], q21[0], q22[0], x1, y1, x2, y2, x, y);

            dst[(j * dw + i)] = yp;
        }
    }
}

void preproc(float* f, float* cos, float* dst, int w,  int h)
{
    const float eps = 1e-5;
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++)
            dst[y * w + x] = log(float(f[y * w + x]) + 1);

    float avg = 0;
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++)
            avg += dst[y * w + x];
    avg = avg / (w * h);

    float sd = 0;
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++)
            sd += (dst[y * w + x] - avg) * (dst[y * w + x] - avg);
    sd = sqrt(sd / (w * h));

    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            dst[y * w + x] = ((dst[y * w + x] - avg) / (sd + eps)) * cos[y * w + x];
        }
    }
}

float*cw, *g, *G, *Ai, *Bi;

void mosse_init(char* src, int srcw, int srch, Rect r)
{
    int cx = r.x + r.w / 2;
    int cy = r.y + r.h / 2;

    // Cosine window
    pu.startTick("cos2d");
    cos2d(cw, r.w, r.h);
    pu.stopTick("cos2d");
    dump2text("dump.cpp.cos2d.txt", cw, r.w, r.h);

    // Gaussian target
    pu.startTick("guassian2d");
    guassian2d(g, r.w, r.h);
    pu.stopTick("guassian2d");
    dump2text("dump.cpp.guassian2d.txt", g, r.w, r.h);

    // DFT Gaussian target
    pu.startTick("Gauss-DFT");
    dft2d(r.w, r.h, g, G);
    pu.stopTick("Gauss-DFT");
    dump2text("dump.cpp.Gauss-DFT.txt", G, 2 * r.w, r.h);

    // load original ROI
    float* roi = new float[r.w * r.h];
    for (size_t y = 0; y < r.h; y++) {
        for (size_t x = 0; x < r.w; x++) {
            roi[y * r.w + x] = (float(src[(y+r.y)*srcw + (x+r.x)]))/255;
        }
    }

    float* fa = new float[r.w * r.h];
    float* fi = new float[r.w * r.h];
    float* Fi = new float[2 * r.w * r.h];
    float angles[8] = { 0, -4.7, 3.8, -4.1, -0.9, 3.0, 0.5, -4.8 };
    for (size_t i = 0; i < 8; i++) {
        float d = angles[i] * (PI / 180);
        float m[3][3] = {
            { cos(d), sin(d), 0 },
            {-sin(d), cos(d), 0 },
            {      0,      0, 1 }
        };

        memset(fa, 0, sizeof(float) * r.w * r.h);

        pu.startTick("affine");
        affine(roi, r.w, r.h, fa, r.w, r.h, m);
        pu.stopTick("affine");

        //dump2yuv(fa, r.w, r.h, i);

        pu.startTick("preproc");
        preproc(fa, cw, fi, r.w, r.h);
        pu.stopTick("preproc");

        pu.startTick("fi-dft2d");
        dft2d(r.w, r.h, fi, Fi);
        pu.stopTick("fi-dft2d");

        pu.startTick("Ai-Bi");
        for (size_t y = 0; y < r.h; y++) {
            for (size_t x = 0; x < r.w; x++) {
                float gr = G[y * r.w * 2 + x * 2];
                float gi = G[y * r.w * 2 + x * 2 + 1];
                float fir = Fi[y * r.w * 2 + x * 2];
                float fii = Fi[y * r.w * 2 + x * 2 + 1];
                Ai[y * r.w * 2 + x * 2] += gr * fir + gi * fii;
                Ai[y * r.w * 2 + x * 2 + 1] += gr * fii + gi * fir;
                Bi[y * r.w * 2 + x * 2] += gr * gr + gi * gi;
                Bi[y * r.w * 2 + x * 2 + 1] += 2 * gr * gi;
            }
        }
        pu.stopTick("Ai-Bi");
    }

    dump2text("dump.cpp.Ai.txt", Ai, 2 * r.w, r.h);
    dump2text("dump.cpp.Bi.txt", Bi, 2 * r.w, r.h);

    delete[] roi, fi, fa, Fi;
}

void mosse_update(char* src, int srcw, int srch, Rect r)
{
    // Hi = Ai / Bi
    float* Hi = new float[2 * r.w * r.h];
    for (size_t y = 0; y < r.h; y++) {
        for (size_t x = 0; x < r.w; x++) {
            float a = Ai[y * 2 * r.w + 2 * x + 0];
            float b = Ai[y * 2 * r.w + 2 * x + 1];
            float c = Bi[y * 2 * r.w + 2 * x + 0];
            float d = Bi[y * 2 * r.w + 2 * x + 1];
            Hi[y * 2 * r.w + 2 * x + 0] = (a * c + b * d) / (c * c + d * d);
            Hi[y * 2 * r.w + 2 * x + 1] = (b * c - a * d) / (c * c + d * d);
        }
    }

    // Fi
    float* fi = new float[r.w * r.h];
    for (size_t y = 0; y < r.h; y++) {
        for (size_t x = 0; x < r.w; x++) {
            fi[y * r.w + x] = (float(src[(y + r.y) * srcw + (x + r.x)])) / 255;
        }
    }
    float* fip = new float[r.w * r.h];
    preproc(fi, cw, fip, r.w, r.h);
    float* Fi = new float[2 * r.w * r.h];
    dft2d(r.w, r.h, fip, Fi);

    // Gi = Hi * Fi
    float* Gi = new float[2 * r.w * r.h];
    for (size_t y = 0; y < r.h; y++) {
        for (size_t x = 0; x < r.w; x++) {
            float a = Hi[y * 2 * r.w + 2 * x + 0];
            float b = Hi[y * 2 * r.w + 2 * x + 1];
            float c = Fi[y * 2 * r.w + 2 * x + 0];
            float d = Fi[y * 2 * r.w + 2 * x + 1];
            Gi[y * 2 * r.w + 2 * x + 0] = a*c - b*d;
            Gi[y * 2 * r.w + 2 * x + 1] = a*d + b*c;
        }
    }

    // gi = IDFT(Gi)
    float* gi = new float[2 * r.w * r.h];
    dft2d(r.w, r.h, Gi, gi);

    float mx = 0, my = 0, max = gi[0];
    for (size_t y = 0; y < r.h; y++) {
        for (size_t x = 0; x < r.w; x++) {
            if (gi[y * 2 * r.w + 2 * x + 0] > max) {
                max = gi[y * 2 * r.w + 2 * x + 0];
                mx = x;
                my = y;
            }
        }
    }

    printf("INFO: mx = %d, my = %d\n", mx, my);

    delete[] Hi, fi, fip, Fi, Gi, gi;

}

void test_cos2d()
{
    const int w = 100, h = 100;
    float* cos = new float[h * w];

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
    float* guass = new float[h * w];

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
    float* x = new float[N];
    float* w = new float[N];
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
    float* f = new float[w * h];
    float* F = new float[w * h];
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

void test_preproc()
{
    const int w = 20, h = 10;
    float* src = new float[w * h];
    float* dst = new float[w * h];
    float* cos = new float[w * h];

    for (size_t i = 0; i < w*h; i++) {
        src[i] = i;
    }

    cos2d(cos, w, h);
    preproc(src, cos, dst, w, h);
    dump2text("out.preproc.txt", dst, w, h);

    delete[] src;
    delete[] dst, cos;
}

void test_affine()
{
    int srcw = 320, srch = 240;
    uint8_t* src = new uint8_t[srcw * srch];
    memset(src, 0, srcw * srch);
    float* srcf = new float[srcw * srch];
    memset(srcf, 0, sizeof(float) * srcw * srch);
    float* dst = nullptr;
    int dstw = 0, dsth = 0;

    ifstream infile;
    infile.open("test.yuv", ios::binary);
    infile.read((char*)src, srcw * srch);
    infile.close();
    for (size_t y = 0; y < srch; y++) {
        for (size_t x = 0; x < srcw; x++) {
            srcf[y * srcw + x] = (float(src[y * srcw + x])) / 255; 
        }
    }

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
    dst = new float[dstw * dsth];
    affine(srcf, srcw, srch, dst, dstw, dsth, translation);
    //dump2yuv(dst, dstw, dsth, 1);
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
    dst = new float[dstw * dsth];
    memset(dst, 0, sizeof(float) * dstw * dsth);
    affine(srcf, srcw, srch, dst, dstw, dsth, scale);
    //dump2yuv(dst, dstw, dsth, 2);
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
    dst = new float[dstw * dsth];
    memset(dst, 0, sizeof(float) * dstw * dsth);
    affine(srcf, srcw, srch, dst, dstw, dsth, shear);
    //dump2yuv(dst, dstw, dsth, 3);
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
    dst = new float[dstw * dsth];
    memset(dst, 0, sizeof(float) * dstw * dsth);
    affine(srcf, srcw, srch, dst, dstw, dsth, rotation);
    //dump2yuv(dst, dstw, dsth, 4);
    delete[] dst;

    delete[] src;
}

void test_mosse()
{
    int srcw = 640, srch = 360;
    Rect rect = { 387, 198, 30, 62 };

    char* src = new char[srcw * srch];
    memset(src, 0, srcw * srch);

    ifstream infile;
    infile.open("tmp1.yuv", ios::binary);
    infile.read(src, srcw * srch);
    infile.close();
    //dump2yuv(src, srcw, srch, 1);

    cw = new float[rect.w * rect.h];
    g = new float[rect.w * rect.h];
    G = new float[2 * rect.w * rect.h];
    Ai = new float[2 * rect.w * rect.h];
    Bi = new float[2 * rect.w * rect.h];
    memset(cw, 0, sizeof(float) * rect.w * rect.h);
    memset(g, 0, sizeof(float) * rect.w * rect.h);
    memset(G, 0, sizeof(float) * 2 * rect.w * rect.h);
    memset(Ai, 0, sizeof(float) * 2 * rect.w * rect.h);
    memset(Bi, 0, sizeof(float) * 2 * rect.w * rect.h);

    mosse_init(src, srcw, srch, rect);

    char* src2 = new char[srcw * srch];
    infile.open("tmp2.yuv", ios::binary);
    infile.read(src, srcw * srch);
    infile.close();

    mosse_update(src2, srcw, srch, rect);

    delete[] cw, g, G, Ai, Bi;
    delete[] src, src2;
    return;
}

int main(int argc, int** argv) 
{
    test_mosse();

    printf("\ndone\n");
    return 0;
}
