#pragma once
#include "math.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

struct RoiRect {
    int x;
    int y;
    int w;
    int h;
};

class Mosse
{
public:
    Mosse();
    ~Mosse();

    int init(char* frame, int pw, int ph, const RoiRect r);
    int update(char* frame, int pw, int ph);

    void dump2txt();
    void dump2bin();

private:
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int picW = 0;
    int picH = 0;
    int dumpIndex = 0;
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
#if USE_OPENCV
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
};

