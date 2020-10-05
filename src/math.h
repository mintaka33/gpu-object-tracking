#pragma once

#define PI 3.14159265

void hanning(const int m, double* d);
void cosWindow(double* cos, const int w, const int h);
void guassian2d(double* guass, const int w, const int h);
void dft2d(const int M, const int N, double* f, double* F);


