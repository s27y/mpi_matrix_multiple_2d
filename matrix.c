/*
 *  mmult.c: matrix multiplication using MPI.
 * There are some simplifications here. The main one is that matrices B and C 
 * are fully allocated everywhere, even though only a portion of them is 
 * used by each processor (except for processor 0)
 */

#include <mpi.h>
#include <stdio.h>
#include <math.h>


void init_matrix(double* a, int row, int col)
{
	int i;

	for(i=0; i<row*col; i++)
	{
		a[i] = 0.0;
	}
}

void multiply_matrix(double* a, double* b, double* c,
 int n,int y,int m)
{
	double tmp;
	int k,i,j;
	for(k = 0; k < y; k++) {
    for(i = 0; i < n; i++) {
        tmp = a[i*y+k];
        for(j = 0; j < m; j++) {
            c[i*n+j] = c[i*n+j] + tmp * b[k*m+j];
        }
    }
	}

}

void print_matrix(double* m, int row, int col)
{
  int i, j = 0;
  for (i=0; i<row; i++) {
	printf("\n\t| ");
	for (j=0; j<col; j++)
	  printf("%2f ", m[i*col+j]);
	printf("|");
  }
  printf("\n");
}


int main(int argc, char *argv[])
{
	double* a;
	double* b;
	double* c;

  double* a_all;
  double* b_all;
  double* c_all;

  double start_time, end_time;
  int my_rank, num_procs, n;
  int myn;
  int i,j;

  int my_start_row,my_start_col;
	/* Initialize MPI environment */ 
	MPI_Init(&argc, &argv);
	/* Get rank of each MPI process */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Get overall number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	{ /* Unnamed validation block */

	if ( argc < 2 ) {
	if (!my_rank)
	   printf("Usage: mpirun -n <num_procs> %s <n>\n", argv[0]);
	MPI_Finalize();
	exit(0);
	}

		  /* Read n from command line argument */
		  n = atoi(argv[1]);

		  if ( n < num_procs ) {
		if ( !my_rank )
		   printf("n should be greated than the number of processes\n");
		MPI_Finalize();
		exit(0);
		  }

		  if ( n % num_procs ) {
		MPI_Finalize();
		if ( !my_rank )
		   printf("n should divide the number of processes\n");
		exit(0);  
		  }

	} /* End of the validation block */
	  myn = n/sqrt(num_procs);

	  a = malloc(myn*myn*sizeof(double));
	  a_all =malloc(n*n*sizeof(double));
	  b = malloc(myn*myn*sizeof(double));
	  b_all =malloc(n*n*sizeof(double));
	  c = malloc(myn*myn*sizeof(double));
	  MPI_Barrier(MPI_COMM_WORLD);

	  my_start_row = my_rank*myn/n;
	  my_start_col = (my_rank*myn)%n;
	  printf("myrank %d startrow %d startcol%d myn%d\n", my_rank,my_start_row,my_start_col,myn);

	  //init_matrix a
	  for(i=0; i<myn*myn; i++)
		{
			a[i] = 1;
			b[i] = 1;
		}


	MPI_Barrier(MPI_COMM_WORLD);
	//gather the whole row for a, whole colom for b
	for(i=0;i<num_procs;i++)
	{
		MPI_Gather(a, myn*myn, MPI_DOUBLE, a_all, myn*myn, MPI_DOUBLE, i, MPI_COMM_WORLD);
		MPI_Gather(b, myn*myn, MPI_DOUBLE, b_all, myn*myn, MPI_DOUBLE, i, MPI_COMM_WORLD);
	}
	

	//print_matrix(a,myn,myn);
	
	{
		print_matrix(a_all,n,n);
		print_matrix(b_all,n,n);
	}

	//print_matrix(&a_all[(my_rank/myn)*n],myn,n);
	//print_matrix(&b_all[(my_rank*myn)*n],n,myn);

	MPI_Barrier(MPI_COMM_WORLD);

		print_matrix(&a_all[(my_rank*myn/n)/n],myn,n);
		print_matrix(&b_all[(my_rank*myn/n)/n],n,myn);
		multiply_matrix(&a_all[(my_rank*myn/n)/n], &b_all[(my_rank*myn/n)/n], c, myn,n,myn);

		print_matrix(c,myn,myn);



	if(my_rank == 0)
	{
		c_all = malloc(n*n*sizeof(double));
		
	}
	MPI_Gather(c, myn*myn, MPI_DOUBLE, c_all, myn*myn, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(my_rank == 0)
	print_matrix(c_all,n,n);

	MPI_Finalize();
	exit(0);

}




