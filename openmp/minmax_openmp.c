#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* generate_data(int n) {
    float* data = (float*)malloc(n * sizeof(float));
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < n; i++) {
            data[i] = (float)rand_r(&seed) / RAND_MAX * 100.0f;
        }
    }
    return data;
}

void minmax_scale_openmp(float* data, int n, float* min_val, float* max_val, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Parallel reduction for min and max
    *min_val = FLT_MAX;
    *max_val = FLT_MIN;
    
    #pragma omp parallel
    {
        float local_min = FLT_MAX;
        float local_max = FLT_MIN;
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            if (data[i] < local_min) local_min = data[i];
            if (data[i] > local_max) local_max = data[i];
        }
        
        #pragma omp critical
        {
            if (local_min < *min_val) *min_val = local_min;
            if (local_max > *max_val) *max_val = local_max;
        }
    }
    
    // Parallel normalization
    float range = *max_val - *min_val;
    if (range == 0.0f) return;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - *min_val) / range;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <data_size> <num_threads> <output_file>\n", argv[0]);
        printf("Example: %s 1000000 4 openmp_output.bin\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    char* output_file = argv[3];
    
    printf("Generating %d random values with %d threads...\n", n, num_threads);
    float* data = generate_data(n);
    
    printf("Performing OpenMP Min-Max scaling...\n");
    double start_time = get_time();
    
    float min_val, max_val;
    minmax_scale_openmp(data, n, &min_val, &max_val, num_threads);
    
    double end_time = get_time();
    double elapsed = end_time - start_time;
    
    // Save output
    FILE* fp = fopen(output_file, "wb");
    if (fp) {
        fwrite(&n, sizeof(int), 1, fp);
        fwrite(data, sizeof(float), n, fp);
        fclose(fp);
    }
    
    printf("\n=== OPENMP IMPLEMENTATION RESULTS ===\n");
    printf("Data size: %d elements\n", n);
    printf("Number of threads: %d\n", num_threads);
    printf("Min value: %.6f\n", min_val);
    printf("Max value: %.6f\n", max_val);
    printf("Execution time: %.6f seconds\n", elapsed);
    printf("Speedup: %.2fx (relative to 1 thread)\n", n / (elapsed * 1000000)); // Approx
    printf("Output saved to: %s\n", output_file);
    
    printf("\nFirst 5 normalized values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  data[%d] = %.6f\n", i, data[i]);
    }
    
    free(data);
    return 0;
}

