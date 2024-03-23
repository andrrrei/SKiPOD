#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (4096 + 2)
double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double A[N][N], B[N][N];

int num_procs, rank, step, start_index, end_index;
MPI_Request requests[4];

void relax();
void init();
void verify();

int main(int an, char **as)
{
    MPI_Init(&an, &as);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    start_index = (rank) * N / num_procs;
    end_index = (rank + 1) * N / num_procs;
    step = end_index - start_index;

    double start_time = MPI_Wtime();

    int it;
    init();
    for (it = 1; it <= itmax; it++)
    {
        eps = 0.;
        relax();
        if (eps < maxeps)
            break;
    }
    verify();

    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    if (rank == 0)
    {
        printf("Processes: %i\nTime: %f seconds\n", num_procs, execution_time);
    }
    MPI_Finalize();
    return 0;
}


void init()
{
    int offset_l = 1, offset_r = 1;
    if (rank == 0)
        offset_l = 0;

    if (rank == num_procs - 1)
        offset_r = 0;

    for (i = start_index - offset_l; i < end_index + offset_r; i++)
        for (j = 0; j <= N - 1; j++)
        {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
                A[i][j] = 0.;
            else
                A[i][j] = (4. + i + j);
        }

}
// копирование граничных значений для соседних процессов
void share_end_rows()
{
    if (rank != 0)
        MPI_Irecv(A[start_index - 1], 1 * N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, requests);

    if (rank != num_procs - 1)
        MPI_Isend(A[end_index - 1], 1 * N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, requests + 2);
}

void share_start_rows()
{
    if (rank != num_procs - 1)
        MPI_Irecv(A[end_index], 1 * N, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, requests + 3);

    if (rank != 0)
        MPI_Isend(A[start_index], 1 * N, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, requests + 1);
}

void waitall()
{
    int count = 4, shift = 0;

    if (rank == 0)
    {
        count = 2;
        shift = 2;
    }
    if (rank == num_procs - 1)
    {
        count = 2;
    }

    MPI_Waitall(count, requests + shift, MPI_STATUSES_IGNORE);
}

void relax()
{
    int offset_l = 0, offset_r = 0;
    if (rank == 0)
        offset_l = 1;

    if (rank == num_procs - 1)
        offset_r = 1;

    for (i = start_index + offset_l; i < end_index - offset_r; i++)
        for (j = 1; j <= N - 2; j++)
        {
            B[i][j] = (A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
        }

    double local_eps = eps;
    for (i = start_index + offset_l; i < end_index - offset_r; i++)
        for (j = 1; j <= N - 2; j++)
        {
            double e;
            e = fabs(A[i][j] - B[i][j]);
            A[i][j] = B[i][j];
            local_eps = Max(local_eps, e);
        }

    if (num_procs != 1)
    {
        share_end_rows();
        share_start_rows();
        waitall();
    }

    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void verify()
{
    double s = 0.;
    double local_s = 0.;
    for (i = start_index; i < end_index; i++)
        for (j = 0; j <= N - 1; j++)
        {
            local_s = local_s + A[i][j]* (i + 1) * (j + 1) / (N * N);
        }

    MPI_Reduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("  S = %f\n", s);
    }
}