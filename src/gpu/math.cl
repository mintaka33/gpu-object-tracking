
#define PI 3.1415926
#define SIGMA 2.0

#define KERNEL_LOG 1

__kernel void hanning(__global double *out, int m) {
  int i = get_global_id(0);
  int size = get_global_size(0);

  out[i] = 0.5 - 0.5 * cos(2 * PI * i / (m - 1));
}

__kernel void cosine2d(__global double *cos2d, __global double *cosw,
                       __global double *cosh, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int i = y * w + x;

  cos2d[i] = sqrt(cosw[x] * cosh[y]);
}

__kernel void gauss2d(__global double *guass, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  double hw = ((double)w) / 2.0;
  double hh = ((double)h) / 2.0;
  double dx = (double)x - hw;
  double dy = (double)y - hh;
  double ep = (dx * dx + dy * dy) / ((double)(SIGMA * SIGMA));

  if (x <= 0 && y <= 0)
    printf("**** kernel-log: w = %d, h = %d\n", w, h);

  guass[y * w * 2 + 2 * x] = exp(-0.5 * ep);     // real part
  guass[y * w * 2 + 2 * x + 1] = 0; // imaginary  part
}

__kernel void logf(__global uchar *src, __global double *dst, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int i = y * w + x;

  dst[i] = log((double)src[i] / 255.0);
}

__kernel void crop(__global uchar *src, __global uchar *dst, int srcw, int srch,
                   int offset_x, int offset_y, int dstw, int dsth) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  dst[y * dstw + x] = src[srcw * (offset_y + y) + offset_x + x];
}

__kernel void affine(__global uchar *src, __global uchar *dst,
                     __global double *mat, int w, int h) {
  int i = get_global_id(0);
  int j = get_global_id(1);

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

__kernel void preproc(__global uchar *src, __global double *cos2d,
                      __global double *dst, float avg, float std, int w,
                      int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  double eps = 0.00001;
  double a = log((float)src[y * w + x]/255.0); // nornalization
  dst[y * w * 2 + x * 2] = ((a - avg) / (std + eps)) * cos2d[y * w + x];
  dst[y * w * 2 + x * 2 + 1] = 0;
}

__kernel void calcH(__global double *G, __global double *F, __global double *H1,
                    __global double *H2, int w, int h) {
  int i = get_global_id(0);
  int j = get_global_id(1);

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

__kernel void correlate(__global double *G, __global double *F,
                        __global double *H1, __global double *H2,
                        __global double *R, int w, int h) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  // calculate H = H1 / H2
  double a = H1[j * 2 * w + 2 * i + 0];
  double b = H1[j * 2 * w + 2 * i + 1];
  double c = H2[j * 2 * w + 2 * i + 0];
  double d = H2[j * 2 * w + 2 * i + 1];
  double temp1 = (a * c + b * d) / (c * c + d * d);
  double temp2 = (b * c - a * d) / (c * c + d * d);

  // R = H * F
  a = temp1;
  b = temp2;
  c = F[j * 2 * w + 2 * i + 0];
  d = F[j * 2 * w + 2 * i + 1];

  R[j * 2 * w + 2 * i + 0] = a * c - b * d;
  R[j * 2 * w + 2 * i + 1] = a * d + b * c;
}
