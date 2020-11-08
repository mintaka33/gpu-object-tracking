
#define PI 3.1415926
#define SIGMA 2.0

__kernel void hanning(__global double* out, int m) 
{
    int i = get_global_id(0);
    int size = get_global_size(0);
    if (i == 0)
        printf("kernel_log: size = %d\n", size);

    out[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
}

__kernel void gauss2d(__global double* guass, int w, int h) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int size_x = get_global_size(0);
    int size_y = get_global_size(1);
    int i = y * w + x;

    if (x == 0 && y == 0)
        printf("kernel_log: size_x = %d, size_y = %d\n", size_x, size_y);

    double hw = ((double)w)/2;
    double hh = ((double)h)/2;
    double dx = (double)x - hw;
    double dy = (double)y - hh;
    double ep = (dx * dx + dy * dy) / ((double)(SIGMA * SIGMA));

    //printf("y = %d, x = %d, i = %d, %f, %f\n", y, x, i, hw, hh);

    guass[i] = exp(-0.5 * ep);;
}
