#pragma once
#include "mosse.h"

const int maxSize = 4096;

inline void freeArray(void *p)
{
    if (!p) {
        delete[] p;
        p = nullptr;
    }
}

template <class T>
T* allocZero(int size) 
{
    T* p = new T[size];
    if (!p)
        return nullptr;

    memset(p, 0, sizeof(T) * size);
    return p;
}

Mosse::Mosse(int w, int h):
    w(w), h(h)
{
    if (w<1 || h<1 || w>maxSize || h>maxSize)
        return;

    int sz = w * h;
    int sz2 = sz * 2; // size of complex number array

    cos = allocZero<double>(sz2);
    G = allocZero<double>(sz2);
    H = allocZero<double>(sz2);
    H1 = allocZero<double>(sz2);
    H2 = allocZero<double>(sz2);

    if (!cos || !G || !H1 || !H2 || !H)
        return;

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

