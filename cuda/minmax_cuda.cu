#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 256
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// CUDA kernel to find min and max
__global__ void find_min_max_kernel(float* data, float* d_min, float* d_max, int n) {
    extern __shared__ float sdata[];
    float* s_min = sdata;
    float* s_max = sdata + blockDim.x;
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    float my_min = FLT_MAX;
    float my_max = FLT_MIN;
    
    while (i < n) {
        if (data[i] < my_min) my_min = data[i];
        if (data[i] > my_max) my_max = data[i];
        i += blockDim.x * gridDim.x;
    }
    
    s_min[tid] = my_min;
    s_max[tid] = my_max;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_min[blockIdx.x] = s_min[0];
        d_max[blockIdx.x] = s_max[0];
    }
}

// CUDA kernel for normalization
__global__ void normalize_kernel(float* data, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float range = max_val - min_val;
    
    if (range == 0.0f) return;
    
    while (idx < n) {
        data[idx] = (data[idx] - min_val) / range;
        idx += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <data_size> <output_file>\n", argv[0]);
        printf("Example: %s 1000000 cuda_output.bin\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    char* output_file = argv[2];
    
    // Allocate host memory
    float* h_data = (float*)malloc(n * sizeof(float));
    
    // Generate random data
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    
    // Allocate device memory
    float* d_data;
    CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Setup CUDA timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // Kernel configuration
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate memory for block results
    float* d_min_blocks, *d_max_blocks;
    float* h_min_blocks = (float*)malloc(num_blocks * sizeof(float));
    float* h_max_blocks = (float*)malloc(num_blocks * sizeof(float));
    
    CHECK(cudaMalloc(&d_min_blocks, num_blocks * sizeof(float)));
    CHECK(cudaMalloc(&d_max_blocks, num_blocks * sizeof(float)));
    
    cudaEventRecord(start);
    
    // Find min and max
    find_min_max_kernel<<<num_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(
        d_data, d_min_blocks, d_max_blocks, n);
    CHECK(cudaGetLastError());
    
    // Copy block results to host
    CHECK(cudaMemcpy(h_min_blocks, d_min_blocks, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_max_blocks, d_max_blocks, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Final reduction on host
    float min_val = FLT_MAX;
    float max_val = FLT_MIN;
    for (int i = 0; i < num_blocks; i++) {
        if (h_min_blocks[i] < min_val) min_val = h_min_blocks[i];
        if (h_max_blocks[i] > max_val) max_val = h_max_blocks[i];
    }
    
    // Normalize
    normalize_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, min_val, max_val, n);
    CHECK(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double elapsed = milliseconds / 1000.0;
    
    // Copy results back to host
    CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Save to file
    FILE* fp = fopen(output_file, "wb");
    if (fp) {
        fwrite(&n, sizeof(int), 1, fp);
        fwrite(h_data, sizeof(float), n, fp);
        fclose(fp);
    }
    
    printf("\n=== CUDA IMPLEMENTATION RESULTS ===\n");
    printf("Data size: %d elements\n", n);
    printf("Blocks: %d, Threads per block: %d\n", num_blocks, BLOCK_SIZE);
    printf("Min value: %.6f\n", min_val);
    printf("Max value: %.6f\n", max_val);
    printf("Execution time: %.6f seconds\n", elapsed);
    printf("Throughput: %.2f million elements/second\n", (n / elapsed) / 1e6);
    printf("Output saved to: %s\n", output_file);
    
    printf("\nFirst 5 normalized values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  data[%d] = %.6f\n", i, h_data[i]);
    }
    
    // Cleanup
    free(h_data);
    free(h_min_blocks);
    free(h_max_blocks);
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_min_blocks));
    CHECK(cudaFree(d_max_blocks));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    return 0;
}
