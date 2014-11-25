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
	  a = malloc(n*n*sizeof(double));
	  a_all =malloc(n*n*sizeof(double));
	  b = malloc(n*n*sizeof(double));
	  c = malloc(n*n*sizeof(double));
	  MPI_Barrier(MPI_COMM_WORLD);

	  my_start_row = my_rank*myn/n;
	  my_start_col = (my_rank*myn)%n;
	  printf("myrank %d startrow %d startcol%d myn%d\n", my_rank,my_start_row,my_start_col,myn);

	  //init_matrix a
	
	  for(i=0; i<myn*n; i++)
		{
			a[i] = 0.0;
			b[i] = 0.0;
		}
	  for(i=0; i<myn; i++)
		{
			for(j=0;j<n;j++)
			{
				if(j>=my_start_col && j<my_start_col+myn)
				{
					a[(i+(my_rank*myn/n)*myn)*n+j] = 1;
					b[(i+(my_rank*myn/n)*myn)*n+j] = 2;
				}
			}
		}


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(a, n*n, MPI_DOUBLE, a_all, n*n, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
	//MPI_Gather(b, my_n, MPI_DOUBLE, b_all, my_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	print_matrix(a_all,n,n);
	//print_matrix(b,n,n);

	MPI_Finalize();
	exit(0);

}




