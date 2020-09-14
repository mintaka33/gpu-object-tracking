// remap.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>

#include <iostream>
#include <fstream>

using namespace std;

float *map1;
float *map2;
char *src;
unsigned char *dst;
int imgw = 1920;
int imgh = 1080;

int loadData()
{
    ifstream file;

    map1 = new float[imgw*imgh];
    file.open("map1.bin", ios::binary);
    if (!file.is_open())
        return -1;
    file.read((char*)map1, sizeof(float)*imgw*imgh);
    file.close();

    map2 = new float[imgw*imgh];
    file.open("map2.bin", ios::binary);
    if (!file.is_open())
        return -1;
    file.read((char*)map2, sizeof(float)*imgw*imgh);
    file.close();

    src = new char[imgw*imgh*3];
    file.open("src.bin", ios::binary);
    if (!file.is_open())
        return -1;
    file.read((char*)src, imgw*imgh*3);
    file.close();

    dst = new unsigned char[imgw * imgh * 3];
    memset(dst, 0, imgw * imgh * 3);

    return 0;
}

void destroy()
{
    if (map1) delete[] map1;
    if (map2) delete[] map2;
    if (src) delete[] src;
    if (dst) delete[] dst;
}

int dumpData(const char* data, int size, const char* fileName)
{
    ofstream file;
    file.open(fileName, ios::binary);
    if (!file.is_open())
    {
        cout << "ERROR: failed to open file!\n";
        return -1;
    }
    file.write(data, size);
    file.close();

    return 0;
}

int copyCrop(const char* src, int srcPitch, char* dst, int dstPitch, int *crop)
{
    if (!src || !dst || !crop)
        return -1;

    for (int y=0; y<crop[3]; y++)
    {
        memcpy_s(dst + y*dstPitch, dstPitch, src + crop[0]*3 + (crop[1] + y)*srcPitch, dstPitch);
    }

    return 0;
}

unsigned char bilinear(float q11, float q12, float q21, float q22, float x1, float y1, float x2, float y2, float x, float y)
{
    float r1, r2, p;
    r1 = (x2 - x)*q11 / (x2 - x1) + (x - x1)*q12 / (x2 - x1);
    r2 = (x2 - x)*q21 / (x2 - x1) + (x - x1)*q22 / (x2 - x1);
    p = (y2 - y)*r1 / (y2 - y1) + (y - y1)*r2 / (y2 - y1);

    if (p < 0) return 0;
    if (p > 255) return 255;
    return (unsigned char)p;
}

void remap(int dstw, int dsth)
{
    for (int h = 0; h < dsth; h++)
    {
        for (int w = 0; w < dstw; w++)
        {
            unsigned char r = 0, g = 0, b = 0;
            float x1, y1, x2, y2, x, y;
            float q11[3], q12[3], q21[3], q22[3];
            int x1i, y1i, x2i, y2i;

            x = map1[h*dstw + w];
            y = map2[h*dstw + w];

            x = (x < 0) ? 0.0 : x;
            y = (y < 0) ? 0.0 : y;
            x = (x > (dstw - 2)) ? (dstw - 2) : x;
            y = (y > (dsth - 2)) ? (dsth - 2) : y;

            x1 = trunc(x); x1i = (int)x1;
            y1 = trunc(y); y1i = (int)y1;
            x2 = x1 + 1; x2i = (int)x2;
            y2 = y1 + 1; y2i = (int)y2;

            q11[0] = (unsigned char)src[(y1i*dstw + x1i) * 3 + 0];
            q12[0] = (unsigned char)src[(y1i*dstw + x2i) * 3 + 0];
            q21[0] = (unsigned char)src[(y2i*dstw + x1i) * 3 + 0];
            q22[0] = (unsigned char)src[(y2i*dstw + x2i) * 3 + 0];
            r = bilinear(q11[0], q12[0], q21[0], q22[0], x1, y1, x2, y2, x, y);

            q11[1] = (unsigned char)src[(y1i*dstw + x1i) * 3 + 1];
            q12[1] = (unsigned char)src[(y1i*dstw + x2i) * 3 + 1];
            q21[1] = (unsigned char)src[(y2i*dstw + x1i) * 3 + 1];
            q22[1] = (unsigned char)src[(y2i*dstw + x2i) * 3 + 1];
            g = bilinear(q11[1], q12[1], q21[1], q22[1], x1, y1, x2, y2, x, y);

            q11[2] = (unsigned char)src[(y1i*dstw + x1i) * 3 + 2];
            q12[2] = (unsigned char)src[(y1i*dstw + x2i) * 3 + 2];
            q21[2] = (unsigned char)src[(y2i*dstw + x1i) * 3 + 2];
            q22[2] = (unsigned char)src[(y2i*dstw + x2i) * 3 + 2];
            b = bilinear(q11[2], q12[2], q21[2], q22[2], x1, y1, x2, y2, x, y);

            dst[(h*dstw + w) * 3 + 0] = r;
            dst[(h*dstw + w) * 3 + 1] = g;
            dst[(h*dstw + w) * 3 + 2] = b;
        }
    }
}

int main()
{
    int dstw = imgw;
    int dsth = imgh;

    if (loadData() != 0)
    {
        cout << "ERROR: load data failed!\n";
        return -1;
    }

    remap(dstw, dsth);

    dumpData((char*)dst, dstw * dsth * 3, "out.rgb");

    int crop[4] = {96, 141, 1727, 810}; // {x, y, width, height}
    int cropSize = crop[2] * crop[3] * 3;
    char* cropBuf = new char[cropSize];
    copyCrop((char*)dst, dstw*3, cropBuf, crop[2]*3, crop);

    dumpData((char*)cropBuf, cropSize, "out2.rgb");

    destroy();
    cout << "execution success!\n";
    return 0;
}

