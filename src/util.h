#pragma once

#include <stdint.h>

void dump2text(char* tag, double* data, const int w, const int h, int i = 0);
void dump2yuv(char* tag, uint8_t* dst, int w, int h, int i = 0);
void double2uchar(uint8_t* dst, double* src, int w, int h);
