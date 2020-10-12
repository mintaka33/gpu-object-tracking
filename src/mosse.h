#pragma once
#include "math.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

struct RoiRect {
    size_t x;
    size_t y;
    size_t w;
    size_t h;
};

class Mosse
{
public:
    Mosse();
    ~Mosse();

    int init(char* frame, int pw, int ph, const RoiRect r);
    int update(char* frame, int pw, int ph, RoiRect& out);

    void dumpResult();
    void dump2txt();
    void dump2bin();

private:
    size_t x = 0;
    size_t y = 0;
    size_t w = 0;
    size_t h = 0;
    size_t picW = 0;
    size_t picH = 0;
    int frameIndex = 0;
    char* curImg = nullptr;
    double* cos = nullptr;
    double* g = nullptr;
    double* f = nullptr;
    double* fa = nullptr;
    double* fi = nullptr;
    double* gi = nullptr;
    double* G = nullptr;
    double* Fi = nullptr;
    double* H1 = nullptr;
    double* H2 = nullptr;
    double* H = nullptr;
    double* Gi = nullptr;

#ifdef USE_OPENCV
    Mat* imgMat = nullptr;
    Mat* cosMat = nullptr;
    Mat* gMat = nullptr;
    Mat* fMat = nullptr;
    Mat* faMat = nullptr;
    Mat* fiMat = nullptr;
    Mat* giMat = nullptr;
    Mat* GMat = nullptr;
    Mat* FiMat = nullptr;
    Mat* H1Mat = nullptr;
    Mat* H2Mat = nullptr;
    Mat* HMat = nullptr;
    Mat* GiMat = nullptr;
#endif

    bool initStatus = false;
    const int affineNum = 8;
    const double rate = 0.125;
};

