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
    Mosse(int w, int h);
    ~Mosse();

    int init(char* frame, const Rect r);
    int update(char* frame, Rect& out);
private:
    int dump() {}

private:
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    char* curImg = nullptr;
    double* cos = nullptr;
    double* g = nullptr;
    double* G = nullptr;
    double* H1 = nullptr;
    double* H2 = nullptr;
    double* H = nullptr;
    bool initStatus = false;
};

