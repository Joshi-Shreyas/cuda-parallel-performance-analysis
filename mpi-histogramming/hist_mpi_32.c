#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_VALUES   2000000
#define DATA_MIN     1
#define DATA_MAX     50000
#define NUM_BINS     32

static int value_to_bin(int v)
{
    long long shifted = (long long)(v - DATA_MIN);
    long long range   = (long long)(DATA_MAX - DATA_MIN + 1);
    int bin = (int)(shifted * NUM_BINS / range);
    if (bin < 0)         bin = 0;
    if (bin >= NUM_BINS) bin = NUM_BINS - 1;
    return bin;
}

int main(int argc, char **argv)
{
    int rank, nprocs, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int *all_data = NULL;
    if (rank == 0) {
        all_data = (int *)malloc(NUM_VALUES * sizeof(int));
        if (!all_data) {
            fprintf(stderr, "malloc failed for all_data\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        srand((unsigned int)time(NULL));
        for (i = 0; i < NUM_VALUES; i++)
            all_data[i] = DATA_MIN + rand() % (DATA_MAX - DATA_MIN + 1);
    }

    int *send_counts = (int *)malloc(nprocs * sizeof(int));
    int *displs      = (int *)malloc(nprocs * sizeof(int));
    if (!send_counts || !displs) {
        fprintf(stderr, "malloc failed for scatter arrays\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base_chunk = NUM_VALUES / nprocs;
    int remainder  = NUM_VALUES % nprocs;
    int offset     = 0;
    for (i = 0; i < nprocs; i++) {
        send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i]      = offset;
        offset        += send_counts[i];
    }

    int local_count = send_counts[rank];
    int *local_data = (int *)malloc(local_count * sizeof(int));
    if (!local_data) {
        fprintf(stderr, "malloc failed for local_data\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t_start = MPI_Wtime();

    MPI_Scatterv(
        all_data, send_counts, displs, MPI_INT,
        local_data, local_count,       MPI_INT,
        0, MPI_COMM_WORLD
    );

    long long local_hist[NUM_BINS];
    for (i = 0; i < NUM_BINS; i++)
        local_hist[i] = 0LL;

    for (i = 0; i < local_count; i++)
        local_hist[value_to_bin(local_data[i])]++;

    long long global_hist[NUM_BINS];
    MPI_Reduce(
        local_hist, global_hist,
        NUM_BINS, MPI_LONG_LONG_INT,
        MPI_SUM, 0, MPI_COMM_WORLD
    );

    double elapsed = MPI_Wtime() - t_start;

    if (rank == 0) {
        double bin_width = (double)(DATA_MAX - DATA_MIN + 1) / NUM_BINS;
        long long total  = 0LL;

        for (i = 0; i < NUM_BINS; i++) {
            int lo = DATA_MIN + (int)(i       * bin_width);
            int hi = DATA_MIN + (int)((i + 1) * bin_width) - 1;
            if (i == NUM_BINS - 1) hi = DATA_MAX;

            printf("Bin %3d  [%6d - %6d] : %7lld\n", i, lo, hi, global_hist[i]);
            total += global_hist[i];
        }

        printf("Total values counted : %lld\n", total);
        printf("Wall-clock time      : %.6f seconds\n", elapsed);

        free(all_data);
    }

    free(local_data);
    free(send_counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
