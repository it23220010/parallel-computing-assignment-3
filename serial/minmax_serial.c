#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

void minmax_serial(float *data, int n) {
    // Step 1: Find min and max
    float min_val = FLT_MAX;
    float max_val = FLT_MIN;
    
    for (int i = 0; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Step 2: Normalize using Min-Max scaling
    float range = max_val - min_val;
    if (range == 0.0f) return;
    
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - min_val) / range;
    }
    
    // Display results
    printf("Serial Min-Max Normalization Results:\n");
    printf("Min value: %.4f\n", min_val);
    printf("Max value: %.4f\n", max_val);
    printf("Range: %.4f\n", range);
    printf("First 3 normalized values: %.4f, %.4f, %.4f\n", 
           data[0], data[1], data[2]);
}

int main() {
    // Fixed array size for proposal
    int n = 10;
    float data[10];
    
    // Initialize with sample data
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    
    printf("Original data (10 values):\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", data[i]);
    }
    printf("\n\n");
    
    // Execute serial min-max normalization
    minmax_serial(data, n);
    
    return 0;
}
