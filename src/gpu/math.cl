
#define PI 3.1415926

__kernel void hanning(__global double* out, int m) 
{
   int i = get_global_id(0);
   out[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
}

__kernel void gauss2d(__global double* guass, int w, int h) 
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int i = y * w + x;
   guass[i] = PI;
}
