#pragma once

#ifdef USE_OPENCV
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
#endif

#define PI 3.14159265

void hanning(const int m, double* d);
void cosWindow(double* cos, const int w, const int h);
void guassian2d(double* guass, const int w, const int h);
void dft2d(const int M, const int N, double* f, double* F);
void idft2d(const int M, const int N, double* F, double* f);
void affine(uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, double m[2][3]);
void preproc(double* f, double* cos, double* dst, int w, int h);

#ifdef USE_OPENCV
void cvAffine(double* src, int sw, int sh, double* dst, int dw, int dh, double m[2][3]);
void cvFFT2d(const size_t w, const size_t h, double* f, double* F);
void cvIFFT2d(const size_t w, const size_t h, double* F, double* f);
#endif

