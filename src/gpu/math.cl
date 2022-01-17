
#define PI 3.1415926
#define SIGMA 2.0

#define KERNEL_LOG 1

__kernel void hanning(__global double *out, int m) {
  int i = get_global_id(0);
  int size = get_global_size(0);

#if KERNEL_LOG
  if (i == 0)
    printf("kernel_log:hanning: size = %d, m = %d\n", size, m);
#endif

  out[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
}

__kernel void cosine2d(__global double *cos2d, __global double *cosw,
                       __global double *cosh, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);
  int i = y * w + x;

#if KERNEL_LOG
  if (x == 0 && y == 0)
    printf("kernel_log:cosine2d: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, w, h);
#endif

  cos2d[i] = sqrt(cosw[x] * cosh[y]);
}

__kernel void gauss2d(__global double *guass, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (x == 0 && y == 0)
    printf("kernel_log:gauss2d: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, w, h);
#endif

  double hw = ((double)w) / 2;
  double hh = ((double)h) / 2;
  double dx = (double)x - hw;
  double dy = (double)y - hh;
  double ep = (dx * dx + dy * dy) / ((double)(SIGMA * SIGMA));

  guass[y * w * 2 + 2 * x] = exp(-0.5 * ep); // real part
  guass[y * w * 2 + 2 * x + 1] = exp(-0.5 * ep); // imaginary  part
}

__kernel void logf(__global uchar *src, __global double *dst, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);
  int i = y * w + x;

#if KERNEL_LOG
  if (x == 0 && y == 0)
    printf("kernel_log:logf: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, w, h);
#endif

  dst[i] = log((double)src[i] / 255.0);
}

__kernel void crop(__global uchar *src, __global uchar *dst, int srcw, int srch,
                   int offset_x, int offset_y, int dstw, int dsth) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (x == 0 && y == 0)
    printf("kernel_log:crop: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, dstw, dsth);
#endif

  dst[y * dstw + x] = src[srcw * (offset_y + y) + offset_x + x];
}

__kernel void affine(__global uchar *src, __global uchar *dst,
                     __global double *mat, int w, int h) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (i == 0 && j == 0) {
    printf("kernel_log:affine: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, w, h);
    printf("kernel_log:affine: matrix = \n %f, %f, %f, \n %f, %f, %f\n", mat[0],
           mat[1], mat[2], mat[3], mat[4], mat[5]);
  }
#endif

  double yp = 0;
  double x1, y1, x2, y2, x, y;
  double q11, q12, q21, q22;
  int x1i, y1i, x2i, y2i;

  x = mat[0] * i + mat[1] * j + mat[2] * 1;
  y = mat[3] * i + mat[4] * j + mat[5] * 1;

  x = (x < 0) ? 0.0 : x;
  y = (y < 0) ? 0.0 : y;
  x = (x > (w - 2)) ? (w - 2) : x;
  y = (y > (h - 2)) ? (h - 2) : y;

  x1 = floor(x);
  y1 = floor(y);
  x1i = (int)x1;
  y1i = (int)y1;

  x2 = x1 + 1;
  y2 = y1 + 1;
  x2i = (int)x2;
  y2i = (int)y2;

  q11 = src[(y1i * w + x1i)];
  q12 = src[(y1i * w + x2i)];
  q21 = src[(y2i * w + x1i)];
  q22 = src[(y2i * w + x2i)];

  // bi-linear interpolation
  double r1, r2, p;
  r1 = (x2 - x) * q11 / (x2 - x1) + (x - x1) * q12 / (x2 - x1);
  r2 = (x2 - x) * q21 / (x2 - x1) + (x - x1) * q22 / (x2 - x1);
  p = (y2 - y) * r1 / (y2 - y1) + (y - y1) * r2 / (y2 - y1);

  dst[(j * w + i)] = p;
}

static float atomic_cmpxchg_f32(volatile __global float *p, float cmp,
                                float val) {
  union {
    unsigned int u32;
    float f32;
  } cmp_union, val_union, old_union;

  cmp_union.f32 = cmp;
  val_union.f32 = val;
  old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *)p,
                                 cmp_union.u32, val_union.u32);
  return old_union.f32;
}

static float atomic_add_f32(volatile __global float *p, float val) {
  float found = *p;
  float expected;
  do {
    expected = found;
    found = atomic_cmpxchg_f32(p, expected, expected + val);
  } while (found != expected);
  return found;
}

__kernel void preproc_sum(__global uchar *src, int w, int h,
                          __global int *sum) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (x == 0 && y == 0) {
    printf("kernel_log:preproc_sum: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, w, h);
  }
#endif

  // the mimic of atomic_add_float is too costly to use
  // atomic_add_f32(sum, log((float)(src[y*w+x])));

  // use build-in atomic_add int, there might be loss due to rounding float to
  // int
  float a = log((float)src[y * w + x]);
  atomic_add(sum, (int)round(a));
}

__kernel void preproc_std(__global uchar *src, __global int *log_sum, int w,
                          int h, __global int *std_sum) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (x == 0 && y == 0) {
    printf("kernel_log:preproc_std: size_x = %d, size_y = %d, w = %d, h = %d\n",
           size_x, size_y, w, h);
    printf("kernel avg = %f\n", (float)(log_sum[0]) / (w * h));
  }
#endif

  float avg = (float)(log_sum[0]) / (w * h);
  float a = log((float)src[y * w + x]);
  float b = (a - avg) * (a - avg);
  atomic_add(std_sum, (int)round(b));
}

__kernel void preproc(__global uchar *src, __global double *cos2d, __global double *dst, float avg, float std, int w, int h)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (x == 0 && y == 0) {
    printf("kernel_log:preproc_std: size_x = %d, size_y = %d, w = %d, h = %d\n", size_x, size_y, w, h);
  }
#endif

  double eps = 0.00001;
  double a = log((float)src[y * w + x]);
  dst[y * w * 2 + x*2] = ((a - avg) / (std + eps)) * cos2d[y * w + x];
  dst[y * w * 2 + x*2 + 1] = 0;
}

__kernel void calcH(__global double *G, __global double *F, __global double *H1, __global double *H2, int w, int h)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int size_x = get_global_size(0);
  int size_y = get_global_size(1);

#if KERNEL_LOG
  if (i == 0 && j == 0) {
    printf("kernel_log:initfilter: size_x = %d, size_y = %d, w = %d, h = %d\n", size_x, size_y, w, h);
  }
#endif

  // H1 += G * np.conj(Fi)
  // H2 += Fi * np.conj(Fi)
  double a = G[j * w * 2 + i * 2 + 0];
  double b = G[j * w * 2 + i * 2 + 1];
  double c = F[j * w * 2 + i * 2 + 0];
  double d = F[j * w * 2 + i * 2 + 1];
  // (a+bi)*(c-di) = (ac + bd) + (bc-ad)i
  H1[j * w * 2 + i * 2 + 0] += a * c + b * d;
  H1[j * w * 2 + i * 2 + 1] += b * c - a * d;
  // (c+di)*(c-di) = (cc+dd)i
  H2[j * w * 2 + i * 2 + 0] += c * c + d * d;
  H2[j * w * 2 + i * 2 + 1] += 0;
}
