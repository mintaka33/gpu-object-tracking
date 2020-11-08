#include <fstream>
#include <stdint.h>

void dump2text(char* tag, double* data, const int w, const int h, int i)
{
    char filename[256] = {};
    sprintf_s(filename, "dump.%04d.%s.%dx%d.txt", i, tag, w, h);
    std::ofstream of(filename);
    char tmp[64] = {};
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            sprintf_s(tmp, "%14.6f", data[x + w * y]);
            of << tmp << ", ";
        }
        of << std::endl;
    }
    of.close();
}

void dump2yuv(char* tag, uint8_t* dst, int w, int h, int i)
{
    std::ofstream outfile;
    char filename[256] = {};
    sprintf_s(filename, "dump.%s.%04d.%dx%d.yuv", tag, i, w, h);
    outfile.open(filename, std::ios::binary);
    outfile.write((char*)dst, w * h);
    outfile.close();
}

void double2uchar(uint8_t* dst, double* src, int w, int h)
{
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            double a = src[j * w + i] * 255;
            dst[j * w + i] = (a > 255) ? 255 : ((a < 0) ? 0 : uint8_t(a));
        }
    }
}
