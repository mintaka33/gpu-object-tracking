#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

struct Rect {
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

    int init(char* frame, int pw, int ph, const Rect r);
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
    double* G = nullptr;
    double* f = nullptr;
    double* fa = nullptr;
    double* fi = nullptr;
    double* Fi = nullptr;
    double* H1 = nullptr;
    double* H2 = nullptr;
    double* H = nullptr;
    double* Gi = nullptr;

    bool initStatus = false;
    const int affineNum = 1;
};

