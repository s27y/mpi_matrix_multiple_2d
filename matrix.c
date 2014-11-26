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

  double* a_local;
  double* b_local;

	double* a_all;
	double* b_all;
  double* c_all;

  double start_time, end_time;
  int my_rank, num_procs, n;
  int myn;
  int i,j;
  MPI_Group word_group;

  int my_start_row,my_start_col;
	/* Initialize MPI environment */ 
	MPI_Init(&argc, &argv);
	/* world group */
	MPI_Comm_group( MPI_COMM_WORLD, &word_group);
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


 		double sqrt_num_procs = sqrt((double)num_procs);
		  if ( floor(sqrt_num_procs) !=  ceil(sqrt_num_procs)) {
		MPI_Finalize();
		   printf("The sqrt of num_procs should be an Integer\n");
		exit(0);  
		  }
		} /* End of the validation block */
	  myn = n/sqrt(num_procs);


	int* ranks;
	MPI_Group* row_groups;
	MPI_Comm* row_coms;
    ranks = malloc(num_procs*sizeof(int));
    row_groups = malloc(sqrt(num_procs)*sizeof(MPI_Group));
    row_coms = malloc(sqrt(num_procs)*sizeof(MPI_Comm));

    	for(i = 0; i < num_procs; i++)
   		{
 	      ranks[i] = i;
 	    }
 	   for(i = 0; i< sqrt(num_procs); i++)
 	   {
 	   	MPI_Group g;
 	   	row_groups[i] = g;
 	   	MPI_Group_incl(word_group, sqrt(num_procs), ranks, &row_groups[i]);

 	   	MPI_Comm com;
 	   	row_coms[i] = com;
 	   	MPI_Comm_create(MPI_COMM_WORLD, row_groups[i], &row_coms[i]);
 	   }
	  //int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup)

	  a = malloc(myn*myn*sizeof(double));
	  a_local = malloc(myn*n*sizeof(double));
	  b = malloc(myn*myn*sizeof(double));
	  b_local = malloc(n*myn*sizeof(double));
	  c = malloc(myn*myn*sizeof(double));
	  MPI_Barrier(MPI_COMM_WORLD);

	 //  //init matrix on root,
	 //  if(my_rank==0)
	 //  {
	 //  	a_all = malloc(n*n*sizeof(double));
	 //  	b_all = malloc(n*n*sizeof(double));

	 //  	for(i=0; i<n*n; i++)
		// {
		// 	a_all[i] = 1.0;
		// 	b_all[i] = 2.0;
		// }
	 //  }
	  

for(i=0; i<myn*myn; i++)
		{
			a[i] = 1.0;
			b[i] = 2.0;
		}

	MPI_Barrier(MPI_COMM_WORLD);

	for(i = 0; i< sqrt(num_procs); i++)
 	   {
 	   	MPI_Group g;
 	   	row_groups[i] = g;
 	   	MPI_Group_incl(word_group, sqrt(num_procs), ranks, &row_groups[i]);

 	   }

 	   
	//MPI_Gather(a, myn*myn, MPI_DOUBLE, a_local, myn*myn, MPI_DOUBLE, s_rank, row_coms[my_rank%myn]);
	//gather the whole row for a, whole colom for b
	for(i=0;i<num_procs;i++)
	{
		//MPI_Gather(a, myn*myn, MPI_DOUBLE, a_local, myn*myn, MPI_DOUBLE, i, MPI_COMM_WORLD);
		//MPI_Gather(b, myn*myn, MPI_DOUBLE, b_local, myn*myn, MPI_DOUBLE, i, MPI_COMM_WORLD);
	}
	

	//print_matrix(a,myn,myn);
	
	{
		//print_matrix(a_local,myn,n);
		//print_matrix(b_local,n,n);
	}

	//print_matrix(&a_local[(my_rank/myn)*n],myn,n);
	//print_matrix(&b_loca[(my_rank*myn)*n],n,myn);

	// MPI_Barrier(MPI_COMM_WORLD);

	// 	//print_matrix(&a_local[(my_rank*myn/n)/n],myn,n);
	// 	//print_matrix(&b_local[(my_rank*myn/n)/n],n,myn);
	// 	multiply_matrix(&a_local[(my_rank*myn/n)/n], &b_local[(my_rank*myn/n)/n], c, myn,n,myn);

	// 	print_matrix(c,myn,myn);



	// if(my_rank == 0)
	// {
	// 	c_all = malloc(n*n*sizeof(double));
	// }
	// MPI_Gather(c, myn*myn, MPI_DOUBLE, c_all, myn*myn, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// if(my_rank == 0)
	// //print_matrix(c_all,n,n);

	MPI_Finalize();
	exit(0);

}




