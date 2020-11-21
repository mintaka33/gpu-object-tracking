
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include <CL/cl.h>

#include "../util.h"

using namespace std;

const string kernel_name = "math.cl";

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

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

    queue = clCreateCommandQueue(context, device, 0, &err);
    CL_CHECK_ERROR(err, "clCreateCommandQueue");
}

void ocl_destroy()
{
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

void gpu_hanning(size_t n, cl_mem &cos1d)
{
    cl_kernel kernel = clCreateKernel(program, "hanning", &err);
    CL_CHECK_ERROR(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cos1d);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 1, sizeof(int), (int*)&n);
    CL_CHECK_ERROR(err, "clSetKernelArg");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &n, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");

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
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, cos2d_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");

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
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");

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
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueNDRangeKernel");

    vector<double> host_log(width * height);
    err = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(double) * width * height, host_log.data(), 0, NULL, NULL);
    CL_CHECK_ERROR(err, "clEnqueueReadBuffer");

    dump2text("log-gpu", host_log.data(), width, height);

    clReleaseKernel(kernel);
}

int main(int argc, char** argv) 
{
    if (argc != 3) {
        printf("ERROR: Invalid command line\n");
        return -1;
    }

    cl_int err;
    size_t width = atoi(argv[1]);
    size_t height = atoi(argv[2]);

    ocl_init();

#if 0
    cl_mem cosw = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_hanning(width, cosw);

    cl_mem cosh = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_hanning(height, cosh);

    cl_mem cos2d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_cos2d(width, height, cosw, cosh, cos2d);

    cl_mem guass2d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width * height, nullptr, &err);
    CL_CHECK_ERROR(err, "clCreateBuffer");
    gpu_gauss2d(width, height, guass2d);
#endif

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

#if 0
    clReleaseMemObject(cosw);
    clReleaseMemObject(cosh);
    clReleaseMemObject(guass2d);
#endif
    ocl_destroy();
    
    return 0;
}
