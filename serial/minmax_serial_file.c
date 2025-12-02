#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <sys/time.h>

// Function to generate random data
float* generate_data(int n) {
    float* data = (float*)malloc(n * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    return data;
}

// Function to write data to file
void write_data(const char* filename, float* data, int n) {
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(&n, sizeof(int), 1, fp);
        fwrite(data, sizeof(float), n, fp);
        fclose(fp);
    }
}

// Function to read data from file
float* read_data(const char* filename, int* n) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    fread(n, sizeof(int), 1, fp);
    float* data = (float*)malloc(*n * sizeof(float));
    fread(data, sizeof(float), *n, fp);
    fclose(fp);
    return data;
}

// Serial Min-Max Scaling
void minmax_scale(float* data, int n, float* min_val, float* max_val) {
    *min_val = FLT_MAX;
    *max_val = FLT_MIN;
    
    // Find min and max
    for (int i = 0; i < n; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
    
    // Normalize
    float range = *max_val - *min_val;
    if (range == 0.0f) return;
    
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - *min_val) / range;
    }
}

// Timer function
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <data_size> <output_file>\n", argv[0]);
        printf("Example: %s 1000000 data_1M.bin\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    char* output_file = argv[2];
    
    printf("Generating %d random float values...\n", n);
    float* data = generate_data(n);
    
    // Save original data for verification
    write_data("original_data.bin", data, n);
    
    printf("Performing serial Min-Max scaling...\n");
    double start_time = get_time();
    
    float min_val, max_val;
    minmax_scale(data, n, &min_val, &max_val);
    
    double end_time = get_time();
    double elapsed = end_time - start_time;
    
    // Save normalized data
    write_data(output_file, data, n);
    
    printf("\n=== SERIAL IMPLEMENTATION RESULTS ===\n");
    printf("Data size: %d elements\n", n);
    printf("Min value: %.6f\n", min_val);
    printf("Max value: %.6f\n", max_val);
    printf("Execution time: %.6f seconds\n", elapsed);
    printf("Throughput: %.2f elements/second\n", n / elapsed);
    printf("Output saved to: %s\n", output_file);
    
    // Verify first 5 values
    printf("\nFirst 5 normalized values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  data[%d] = %.6f\n", i, data[i]);
    }
    
    free(data);
    return 0;
}
