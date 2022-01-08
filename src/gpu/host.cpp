
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <string>
#include <vector>
#include <fstream>

#include <CL/cl.h>

#include "../util.h"

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

    vector<double> host_guass2d(width * height);
    err = clEnqueueReadBuffer(queue, guass2d, CL_TRUE, 0, sizeof(double) * width * height, host_guass2d.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueReadBuffer");

    dump2text("guass2d-gpu", host_guass2d.data(), width, height);

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

void test_gpu_gauss2d(size_t width, size_t height)
{
    cl_mem guass2d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_gauss2d(width, height, guass2d);
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

void test_gpu_crop(size_t x, size_t y, size_t w, size_t h)
{
    size_t srcw = 1920, srch = 1080;
    vector<int8_t> inbuf(srcw * srch, 0);
    init_srcbuf((char*)inbuf.data(), srcw * srch); 

    vector<int8_t> outgpu(w * h, 0);
    cl_mem clsrc = clCreateBuffer(context, CL_MEM_READ_ONLY, srcw * srch, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    err = clEnqueueWriteBuffer(queue, clsrc, CL_TRUE, 0, srcw * srch, inbuf.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueWriteBuffer");
    cl_mem cldst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, w * h, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");

    gpu_crop(clsrc, cldst, srcw, srch, x, y, w, h);

    err = clEnqueueReadBuffer(queue, cldst, CL_TRUE, 0, w * h, outgpu.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueReadBuffer");

    // generate reference
    vector<int8_t> outref(w*h, 0);
    for (size_t j = 0; j < h; j++) {
        for (size_t i = 0; i < w; i++) {
            outref[j * w + i] = inbuf[(y+j)*srcw + (x+i)];
        }
    }
    //ofstream roifile;
    //roifile.open("..\\..\\roi.yuv", ios::binary);
    //roifile.write((const char*)outref.data(), w * h);
    //roifile.close();

    // compare gpu output and reference
    int mismatch_count = 0;
    for (size_t i = 0; i < w*h; i++) {
        if (outgpu[i] != outref[i]) {
            mismatch_count++;
        }
    }
    printf("INFO: test_gpu_crop mismatch_count =%d\n", mismatch_count);

    clReleaseMemObject(clsrc);
    clReleaseMemObject(cldst);
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

    test_gpu_crop(roi.x, roi.y, roi.width, roi.height);

    ocl_destroy();
    
    return 0;
}
