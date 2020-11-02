
#define PROGRAM_FILE "math.cl"
#define KERNEL_FUNC "cos_win"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <vector>

#include <CL/cl.h>

using namespace std;

int main() 
{
    /* Host/device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int i, err;

    /* Program/kernel data structures */
    cl_program program;
    FILE* program_handle;
    size_t program_size, log_size;
    cl_kernel kernel;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't find any platforms");
        exit(1);
    }

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0) {
        perror("Couldn't find any devices");
        exit(1);
    }

    /* Create the context */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    /* Read program file and place content into buffer */
    program_handle = fopen(PROGRAM_FILE, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    vector<char> program_buffer(program_size + 1, 0);
    fread(program_buffer.data(), sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    char* prog_buf = program_buffer.data();
    program = clCreateProgramWithSource(context, 1, (const char**)&prog_buf, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        vector<char> program_log(log_size + 1, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, &program_log[0], NULL);
        printf("%s\n", &program_log[0]);
        exit(1);
    }

    /* Create kernel for the mat_vec_mult function */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create the kernel");
        exit(1);
    }

    size_t width = 200, height = 100;

    /* Create CL buffers to hold input and output data */
    cl_mem cosw = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * width, nullptr, &err);
    if (err < 0) {
        perror("Couldn't create a buffer object");
        exit(1);
    }

    /* Create kernel arguments from the CL buffers */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cosw);
    if (err < 0) {
        perror("Couldn't set the kernel argument");
        exit(1);
    }
    err = clSetKernelArg(kernel, 1, sizeof(int), (int*)&width);
    if (err < 0) {
        perror("Couldn't set the kernel argument");
        exit(1);
    }

    /* Create a CL command queue for the device*/
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        perror("Couldn't create the command queue");
        exit(1);
    }

    /* Enqueue the command queue to the device */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &width, NULL, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't enqueue the kernel execution command");
        exit(1);
    }

    vector<double> host_cosw(width);
    /* Read the result */
    err = clEnqueueReadBuffer(queue, cosw, CL_TRUE, 0, sizeof(double) * width, host_cosw.data(), 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't enqueue the read buffer command");
        exit(1);
    }


    /* Deallocate resources */
    clReleaseMemObject(cosw);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
