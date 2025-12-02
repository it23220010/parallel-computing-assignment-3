#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3 && rank == 0) {
        printf("Usage: mpirun -np <processes> %s <data_size> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    int n = atoi(argv[1]);
    char* output_file = argv[2];
    
    // Calculate local size
    int local_n = n / size;
    if (rank < n % size) {
        local_n++;
    }
    
    // Generate local data
    float* local_data = (float*)malloc(local_n * sizeof(float));
    srand(time(NULL) + rank);
    for (int i = 0; i < local_n; i++) {
        local_data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Find local min and max
    float local_min = FLT_MAX;
    float local_max = FLT_MIN;
    for (int i = 0; i < local_n; i++) {
        if (local_data[i] < local_min) local_min = local_data[i];
        if (local_data[i] > local_max) local_max = local_data[i];
    }
    
    // Global reduction
    float global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    
    // Normalize local data
    float range = global_max - global_min;
    if (range != 0.0f) {
        for (int i = 0; i < local_n; i++) {
            local_data[i] = (local_data[i] - global_min) / range;
        }
    }
    
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    // Gather results to root process
    if (rank == 0) {
        float* all_data = (float*)malloc(n * sizeof(float));
        int* recv_counts = (int*)malloc(size * sizeof(int));
        int* displs = (int*)malloc(size * sizeof(int));
        
        // Gather counts
        int temp_count = local_n;
        recv_counts[0] = local_n;
        displs[0] = 0;
        
        for (int i = 1; i < size; i++) {
            int other_n = n / size;
            if (i < n % size) other_n++;
            recv_counts[i] = other_n;
            displs[i] = displs[i-1] + recv_counts[i-1];
            temp_count += other_n;
        }
        
        MPI_Gatherv(local_data, local_n, MPI_FLOAT,
                   all_data, recv_counts, displs, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
        
        // Save to file
        FILE* fp = fopen(output_file, "wb");
        if (fp) {
            fwrite(&n, sizeof(int), 1, fp);
            fwrite(all_data, sizeof(float), n, fp);
            fclose(fp);
        }
        
        printf("\n=== MPI IMPLEMENTATION RESULTS ===\n");
        printf("Data size: %d elements\n", n);
        printf("Number of processes: %d\n", size);
        printf("Global min: %.6f\n", global_min);
        printf("Global max: %.6f\n", global_max);
        printf("Execution time: %.6f seconds\n", elapsed);
        printf("Output saved to: %s\n", output_file);
        
        printf("\nFirst 5 normalized values:\n");
        for (int i = 0; i < 5 && i < n; i++) {
            printf("  data[%d] = %.6f\n", i, all_data[i]);
        }
        
        free(all_data);
        free(recv_counts);
        free(displs);
    } else {
        MPI_Gatherv(local_data, local_n, MPI_FLOAT,
                   NULL, NULL, NULL, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
    }
    
    free(local_data);
    MPI_Finalize();
    return 0;
}

