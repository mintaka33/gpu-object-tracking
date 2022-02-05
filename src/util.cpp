#include <fstream>
#include <stdint.h>
#include <iomanip>
#include <random>
#include <chrono>

void getMatrix(int w, int h, double* mat)
{
    double r[5] = {};
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-0.1, 0.1);
    for (size_t i = 0; i < 5; i++) {
        r[i] = distribution(generator); // (-0.1, 0.1)
    }

    //srand(time(NULL));
    //for (size_t i = 0; i < 5; i++) {
    //    r[i] = (((double)rand() / (RAND_MAX)) - 0.5) * 0.2; // (-0.1, 0.1)
    //}

    double c = cos(r[0]);
    double s = sin(r[0]);
    double m[2][3] = {};
    m[0][0] = c + r[1];
    m[0][1] = -s + r[2];
    m[1][0] = s + r[3];
    m[1][1] = c + r[4];
    double c1 = w / 2.0, c2 = h / 2.0;
    double t1 = m[0][0] * c1 + m[0][1] * c2;
    double t2 = m[1][0] * c1 + m[1][1] * c2;
    m[0][2] = c1 - t1;
    m[1][2] = c2 - t2;

    mat[0] = m[0][0], mat[1] = m[0][1], mat[2] = m[0][2];
    mat[3] = m[1][0], mat[4] = m[1][1], mat[5] = m[1][2];

    //printf("%10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n", mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]);
}

void dump2text(char* tag, double* data, const int w, const int h, int i)
{
    char filename[256] = {};
    sprintf_s(filename, "dump.%04d.%s.%dx%d.txt", i, tag, w, h);
    std::ofstream of(filename);
    char tmp[64] = {};
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            sprintf_s(tmp, "%+.18e", data[x + w * y]);
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
