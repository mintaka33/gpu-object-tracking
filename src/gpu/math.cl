
#define PI 3.1415926

__kernel void cos_win(__global double* out, int m) 
{
   int i = get_global_id(0);
   out[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
}
