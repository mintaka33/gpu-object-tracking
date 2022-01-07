
#define PI 3.1415926
#define SIGMA 2.0

__kernel void hanning(__global double* out, int m) 
{
    int i = get_global_id(0);
    int size = get_global_size(0);
    if (i == 0)
        printf("kernel_log:hanning: size = %d, m = %d\n", size, m);

    out[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
}

__kernel void cosine2d(__global double* cos2d, __global double* cosw, __global double* cosh, int w, int h) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int size_x = get_global_size(0);
    int size_y = get_global_size(1);
    int i = y * w + x;

    if (x == 0 && y == 0)
        printf("kernel_log:cosine2d: size_x = %d, size_y = %d, w = %d, h = %d\n", size_x, size_y, w, h);

    cos2d[i] = sqrt(cosw[x] * cosh[y]);
}

__kernel void gauss2d(__global double* guass, int w, int h) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int size_x = get_global_size(0);
    int size_y = get_global_size(1);
    int i = y * w + x;

    if (x == 0 && y == 0)
        printf("kernel_log:gauss2d: size_x = %d, size_y = %d, w = %d, h = %d\n", size_x, size_y, w, h);

    double hw = ((double)w)/2;
    double hh = ((double)h)/2;
    double dx = (double)x - hw;
    double dy = (double)y - hh;
    double ep = (dx * dx + dy * dy) / ((double)(SIGMA * SIGMA));

    guass[i] = exp(-0.5 * ep);
}

__kernel void logf(__global uchar* src, __global double* dst, int w,  int h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int size_x = get_global_size(0);
    int size_y = get_global_size(1);
    int i = y * w + x;

    if (x == 0 && y == 0)
        printf("kernel_log:log: size_x = %d, size_y = %d, w = %d, h = %d\n", size_x, size_y, w, h);

    dst[i] = log((double)src[i] / 255.0);
}

__kernel void crop(__global uchar* src, __global uchar* dst, int srcw,  int srch, int offset_x, int offset_y, int dstw, int dsth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int size_x = get_global_size(0);
    int size_y = get_global_size(1);

    if (x == 0 && y == 0)
        printf("kernel_log:log: size_x = %d, size_y = %d, w = %d, h = %d\n", size_x, size_y, dstw, dsth);

    dst[y*dstw+x] = src[srcw*(offset_y+y)+offset_x+x];
}
