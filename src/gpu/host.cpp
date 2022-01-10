
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <string>
#include <vector>
#include <fstream>

#include <CL/cl.h>

#include "../util.h"
#include "../math.h"

#define VKFFT_BACKEND 3
#include "vkFFT.h"

using namespace std;

const string kernel_name = "math.cl";

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_event profile_event;
cl_program program;
size_t timer_res;
cl_ulong time_start, time_end;

typedef struct _ROI {
    size_t x;
    size_t y;
    size_t width;
    size_t height;
} ROI;

static ROI roi = {};
static int g_dump_result = true;

#define CL_CHECK_ERROR(err, msg) \
if (err < 0 ) { \
    printf("ERROR: %s failed with err = %d, in function %s, line %d\n", msg, err, __FUNCTION__, __LINE__); \
    exit(0); \
}

void ocl_init()
{
    FILE* fp;
    size_t program_size, log_size;

    err = clGetPlatformIDs(1, &platform, NULL);
    CL_CHECK_ERROR(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CL_CHECK_ERROR(err, "clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK_ERROR(err, "clCreateContext");

    string kernel_path = ".\\" + kernel_name;
    if (!(fp = fopen(kernel_path.c_str(), "r"))) {
        printf("INFO: Cannot open kernel file %s, trying another path\n", kernel_path.c_str());
        kernel_path = "..\\..\\src\\gpu\\" + kernel_name;
        if (!(fp = fopen(kernel_path.c_str(), "r"))) {
            printf("ERROR: Cannot open kernel file %s\n", kernel_path.c_str());
            exit(1);
        }
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    vector<char> program_buffer(program_size + 1, 0);
    fread(program_buffer.data(), sizeof(char), program_size, fp);
    fclose(fp);

    char* prog_buf = program_buffer.data();
    program = clCreateProgramWithSource(context, 1, (const char**)&prog_buf, &program_size, &err);
    CL_CHECK_ERROR(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        vector<char> program_log(log_size + 1, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, &program_log[0], NULL);
        printf("%s\n", &program_log[0]);
        exit(1);
    }

    clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(timer_res), &timer_res, NULL);
    printf("INFO: Device profiling timer resolution is %lld\n", timer_res);

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK_ERROR(err, "clCreateCommandQueue");
}

void ocl_destroy()
{
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

void print_perf()
{
    clFinish(queue);
    clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    printf("INFO: kernel execution time = %f us\n", (time_end - time_start) / 1000.0);
}

void init_srcbuf(char* buf, int size)
{
    string yuvfile = "..\\..\\test.yuv";
    ifstream infile;
    infile.open(yuvfile.c_str(), ios::binary);
    if (!infile.is_open()) {
        printf("ERROR: failed to open input yuv file %s\n", yuvfile);
        exit(1);
    }
    infile.read(buf, size);
    infile.close();
}

void dump_clbuf(char* tag, cl_mem clbuffer, uint64_t size, int w, int h, int i, bool istext)
{
    cl_int res;
    vector<double> outdata(size, 0);
    res = clEnqueueReadBuffer(queue, clbuffer, CL_TRUE, 0, size, outdata.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");
    clFinish(queue);

    if (istext) {
        dump2text(tag, (double*)outdata.data(), w, h, i);
    } else {
        dump2yuv(tag, (uint8_t*)outdata.data(), w, h, i);
    }
}

void gpu_hanning(size_t n, cl_mem &cos1d)
{
    cl_kernel kernel = clCreateKernel(program, "hanning", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cos1d);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 1, sizeof(int), (int*)&n);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &n, NULL, 0, NULL, &profile_event);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    print_perf();

    vector<double> host_cos1d(n);
    err = clEnqueueReadBuffer(queue, cos1d, CL_TRUE, 0, sizeof(double) * n, host_cos1d.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueReadBuffer");

    clReleaseKernel(kernel);
}

void gpu_cos2d(size_t width, size_t height, cl_mem& cosw, cl_mem& cosh, cl_mem& cos2d)
{
    cl_kernel kernel = clCreateKernel(program, "cosine2d", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cos2d);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cosw);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cosh);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 3, sizeof(int), (int*)&width);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 4, sizeof(int), (int*)&height);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    size_t cos2d_work_size[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, cos2d_work_size, NULL, 0, NULL, &profile_event);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    print_perf();

    vector<double> host_cos2d(width* height);
    err = clEnqueueReadBuffer(queue, cos2d, CL_TRUE, 0, sizeof(double) * width * height, host_cos2d.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueReadBuffer");

    dump2text("cos2d-gpu", host_cos2d.data(), width, height);

    clReleaseKernel(kernel);
}

void gpu_gauss2d(size_t width, size_t height, cl_mem& guass2d)
{
    guass2d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");

    cl_kernel kernel = clCreateKernel(program, "gauss2d", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &guass2d);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 1, sizeof(int), (int*)&width);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 2, sizeof(int), (int*)&height);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    size_t work_size[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_size, NULL, 0, NULL, &profile_event);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    print_perf();

    clReleaseKernel(kernel);
}

void gpu_log(size_t width, size_t height, cl_mem& src, cl_mem& dst)
{
    cl_kernel kernel = clCreateKernel(program, "logf", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(int), (int*)&width);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 3, sizeof(int), (int*)&height);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    size_t work_size = width*height;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size, NULL, 0, NULL, &profile_event);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    print_perf();

    vector<double> host_log(width * height);
    err = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(double) * width * height, host_log.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueReadBuffer");

    dump2text("log-gpu", host_log.data(), width, height);

    clReleaseKernel(kernel);
}

void gpu_crop(cl_mem clsrc, cl_mem cldst, int srcw, int srch, int x, int y, int dstw, int dsth)
{
    cl_kernel kernel = clCreateKernel(program, "crop", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clsrc);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cldst);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(int), (int*)&srcw);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 3, sizeof(int), (int*)&srch);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 4, sizeof(int), (int*)&x);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 5, sizeof(int), (int*)&y);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 6, sizeof(int), (int*)&dstw);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 7, sizeof(int), (int*)&dsth);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    size_t work_size[2] = { dstw, dsth };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_size, NULL, 0, NULL, &profile_event);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    print_perf();

    clReleaseKernel(kernel);
}

void gpu_affine(cl_mem clsrc, cl_mem cldst, int w, int h, double m[2][3])
{
    cl_kernel kernel = clCreateKernel(program, "affine", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    cl_mem clmatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * 2 * 3, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    err = clEnqueueWriteBuffer(queue, clmatrix, CL_TRUE, 0, sizeof(double) * 2 * 3, &m[0][0], 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clsrc);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cldst);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clmatrix);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 3, sizeof(int), (int*)&w);
    CL_CHECK_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 4, sizeof(int), (int*)&h);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    size_t work_size[2] = { w, h };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_size, NULL, 0, NULL, &profile_event);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    print_perf();

    clReleaseMemObject(clmatrix);
    clReleaseKernel(kernel);
}

VkFFTResult gpu_fft(cl_mem clinbuffer, cl_mem& clbuffer, int w, int h, bool r2c)
{
    cl_int res = CL_SUCCESS;
    //zero-initialize configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D
    configuration.size[0] = w;
    configuration.size[1] = h;
    configuration.numberBatches = 0;
    configuration.doublePrecision = 1;
    configuration.performR2C = (r2c)? 1 : 0; // perform R2C/C2R decomposition (0 - off, 1 - on)
    uint64_t num_items = configuration.size[0] * configuration.size[1];

    // out-of-place R2C FFT with custom strides
    uint64_t inputBufferSize = (r2c) ? sizeof(double) * 2 * num_items : sizeof(double) * num_items;
    uint64_t bufferSize = sizeof(double) * 2 * num_items; // (configuration.size[0] / 2 + 1)* configuration.size[1];

    configuration.isInputFormatted = 1;
    configuration.inputBufferStride[0] = configuration.size[0];
    configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * configuration.size[1];
    configuration.bufferStride[0] = configuration.size[0];
    configuration.bufferStride[1] = configuration.bufferStride[0] * configuration.size[1];

    configuration.device = &device;
    configuration.platform = &platform;
    configuration.context = &context;

    // computation buffer in device
    clbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");

    configuration.inputBuffer = &clinbuffer;
    configuration.inputBufferSize = &inputBufferSize;
    configuration.buffer = &clbuffer;
    configuration.bufferSize = &bufferSize;
    //configuration.outputBuffer = &cloutbuffer;
    //configuration.outputBufferSize = &outputBufferSize;

    VkFFTResult resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS && resFFT != VKFFT_ERROR_ENABLED_saveApplicationToString) {
        printf("ERROR: initializeVkFFT failed with resFFT = %d\n", resFFT);
        return resFFT;
    }

    VkFFTLaunchParams launchParams = {};
    launchParams.inputBuffer = &clinbuffer;
    launchParams.buffer = &clbuffer;
    //launchParams.outputBuffer = &cloutbuffer;
    launchParams.commandQueue = &queue;

    // FFT
    resFFT = VkFFTAppend(&app, -1, &launchParams);
    if (resFFT != VKFFT_SUCCESS) {
        printf("ERROR: FFT failed with resFFT = %d\n", resFFT);
        return resFFT;
    }
    clFinish(queue);

    deleteVkFFT(&app);

    return VKFFT_SUCCESS;
}

void test_gpu_cos2d(size_t width, size_t height)
{
    cl_mem cosw = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_hanning(width, cosw);

    cl_mem cosh = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_hanning(height, cosh);

    cl_mem cos2d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_cos2d(width, height, cosw, cosh, cos2d);

    clReleaseMemObject(cosw);
    clReleaseMemObject(cosh);
    clReleaseMemObject(cos2d);
}

void test_gpu_gauss2d(size_t w, size_t h)
{
    cl_mem guass2d;

    gpu_gauss2d(w, h, guass2d);

    dump_clbuf("gpu-guass2d", guass2d, sizeof(double)*w*h, w, h, 0, true);

    clReleaseMemObject(guass2d);
}

void test_gpu_preproc(size_t width, size_t height)
{
    size_t aligned_size = ((width * height + 63) / 64) * 64;
    uint8_t* d = (uint8_t*)_aligned_malloc(sizeof(uint8_t) * aligned_size, 4096);
    memset(d, 0, aligned_size);
    for (size_t i = 0; i < width * height; i++) {
        d[i] = i % 256;
    }
    cl_mem data_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(uint8_t) * width * height, d, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    cl_mem data_log = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_log(width, height, data_in, data_log);

    clReleaseMemObject(data_in);
    clReleaseMemObject(data_log);
}

void crop_roi(size_t w, size_t h, size_t x, size_t y, cl_mem& crop_dst)
{
    size_t srcw = 1920, srch = 1080;
    vector<int8_t> inbuf(srcw * srch, 0);
    init_srcbuf((char*)inbuf.data(), srcw * srch);
    //dump2yuv("srcyuv", (uint8_t *)inbuf.data(), w, h, 0);

    cl_mem crop_src = clCreateBuffer(context, CL_MEM_READ_ONLY, srcw * srch, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    err = clEnqueueWriteBuffer(queue, crop_src, CL_TRUE, 0, srcw * srch, inbuf.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");

    crop_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, w * h, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");

    gpu_crop(crop_src, crop_dst, srcw, srch, x, y, w, h);

    clReleaseMemObject(crop_src);
}

void affine_roi(size_t w, size_t h, cl_mem crop_dst, cl_mem& affine_dst)
{
    affine_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, w * h, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");

    double m[2][3] = {};
    getMatrix(w, h, m[0]);
    printf("host matrix = \n %f, %f, %f \n %f, %f, %f \n", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]);

    gpu_affine(crop_dst, affine_dst, w, h, m);
}

void test_gpu_affine(size_t x, size_t y, size_t w, size_t h)
{
    cl_mem crop_dst;
    cl_mem affine_dst;

    crop_roi(w, h, x, y, crop_dst);
    dump_clbuf("gpu-roi", crop_dst, w*h, w, h, 0, false);

    affine_roi(w, h, crop_dst, affine_dst);
    dump_clbuf("gpu-affine", affine_dst, w * h, w, h, 0, false);

    clReleaseMemObject(crop_dst);
    clReleaseMemObject(affine_dst);
}

void test_gpu_fft(int w, int h)
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    uint64_t num_items = w * h;
    uint64_t inputBufferSize = sizeof(double) * 2 * num_items;
    uint64_t outputBufferSize = sizeof(double) * 2 * num_items;
    uint64_t bufferSize = sizeof(double) * 2 * num_items; 
    vector<double> indata(2 * num_items, 0);
    vector<double> outdata(2 * num_items, 0);
    for (size_t i = 0; i < 2 * num_items; i += 2) {
        indata[i] = i;
    }

    // input buffer in device
    cl_mem clinbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBufferSize, 0, &res);
    CL_CHECK_ERROR(err, "clCreateBuffer");

    res = clEnqueueWriteBuffer(queue, clinbuffer, CL_TRUE, 0, inputBufferSize, indata.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");

    // computation buffer in device
    cl_mem clbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");

    // execute GPU FFT
    printf("gpu-fft: width = %d, height = %d\n", w, h);
    resFFT = gpu_fft(clinbuffer, clbuffer, w, h, true);
    printf("resFFT = % d\n", resFFT);
    dump_clbuf("gpu-fft", clbuffer, bufferSize, w, h, 0, true);

    clReleaseMemObject(clbuffer);
    clReleaseMemObject(clinbuffer);
}

void track_init(const ROI& roi)
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    int w = roi.width;
    int h = roi.height;
    int x = roi.x;
    int y = roi.y;
    cl_mem guass2d;
    cl_mem clbuffer;
    cl_mem crop_dst;
    cl_mem affine_dst;

    // generate gauss distribution
     gpu_gauss2d(w, h, guass2d);

    // execute GPU FFT
    resFFT = gpu_fft(guass2d, clbuffer, w, h, false);
    printf("INFO: gpu_fft return = %d\n", resFFT);
    dump_clbuf("gpu-fft", clbuffer, 2 * w * h, 2*w, h, 0, true);

    // crop the ROI region from source frame
    crop_roi(w, h, x, y, crop_dst);
    dump_clbuf("gpu-roi", crop_dst, w * h, w, h, 0, false);

    // do affine transformation for the ROI region
    affine_roi(w, h, crop_dst, affine_dst);
    dump_clbuf("gpu-affine", affine_dst, w * h, w, h, 0, false);

    clReleaseMemObject(crop_dst);
    clReleaseMemObject(affine_dst);
    clReleaseMemObject(guass2d);
    clReleaseMemObject(clbuffer);
}

void parse_arg(int argc, char** argv)
{
    roi.x = 6;
    roi.y = 599;
    roi.width = 517;
    roi.height = 421;

    switch (argc)
    {
    case 1:
        printf("default: x = %d, y =%d, width = %d, height = %d\n", roi.x, roi.y, roi.width, roi.height);
        break;
    case 3:
        roi.width = atoi(argv[1]);
        roi.height = atoi(argv[2]);
        break;
    case 5:
        roi.width = atoi(argv[1]);
        roi.height = atoi(argv[2]);
        break;
    default:
        printf("ERROR: invalid command line! exit\n");
        exit(1);
        break;
    }
}

int main(int argc, char** argv) 
{
    parse_arg(argc, argv);

    ocl_init();

    //test_gpu_cos2d(roi.width, roi.height);
    //test_gpu_gauss2d(roi.width, roi.height);
    //test_gpu_preproc(roi.width, roi.height);
    //test_gpu_affine(roi.x, roi.y, roi.width, roi.height);
    //test_gpu_fft(roi.width, roi.height);

    track_init(roi);

    ocl_destroy();
    
    return 0;
}
