#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#define  N   (64+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;

double A [N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
	int p[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160};
    double time_start;
    double time_end;
    int it;

    for(i = 0; i < 18; i++) { 
        omp_set_num_threads(p[i]); //устанавливаем количество нитей
        time_start = omp_get_wtime();
        init();
        for(it=1; it<=itmax; it++)
        {
            eps = 0.;
            relax();
            if (eps < maxeps) break;
        }
        verify();
        time_end = omp_get_wtime();
        printf("threads: %d N: %d  time: %10f  \n\n", p[i], N, time_end - time_start);
    }

	return 0;
}


void init()
{ 
#pragma omp parallel private(j) shared(A)
{
#pragma omp single
{
	for(j=0; j<=N-1; j++)
#pragma omp task firstprivate(i)
{
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1)
		A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
}
}
}
} 


void relax() {
    int i, j;
    double e;
    #pragma omp parallel for private(i, j, e) shared(A) reduction(max: eps)
    for (i = 1; i <= N - 2; i++) {
        for (j = (i % 2) + 1; j <= N - 2; j += 2) {
            e = A[i][j];
            A[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.0;
            eps = Max(eps, fabs(e - A[i][j]));
        }
    }
    #pragma omp barrier
    #pragma omp parallel for private(i, j, e) shared(A) reduction(max: eps)
    for (i = 1; i <= N - 2; i++) {
        for (j = 2 - (i % 2); j <= N - 2; j += 2) {
            e = A[i][j];
            A[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.0;
            eps = Max(eps, fabs(e - A[i][j]));
        }
    }
}



void verify()
{ 
	double s;

	s=0.;
    #pragma omp parallel shared(A) private(j) reduction(+:s)
    {
        for(j=0; j<=N-1; j++)
        #pragma omp for private(i) nowait schedule(auto)
        for(i=0; i<=N-1; i++)
        {
            s=s+A[i][j]*(i+1)*(j+1)/(N*N);
        }
    }
	printf("  S = %f\n",s);

}
