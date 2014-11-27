/*
 *  mmult.c: matrix multiplication using MPI.
 * There are some simplifications here. The main one is that matrices B and C 
 * are fully allocated everywhere, even though only a portion of them is 
 * used by each processor (except for processor 0)
 */

#include <mpi.h>
#include <stdio.h>
#include <math.h>

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

 void multiply_matrix(double* a, double* b, double* c,
 	int n,int y,int m,int my_rank)
 {
 	double tmp;
 	int k,i,j;
 	int row_offset, col_offset;
 	for(k = 0; k < y; k++) {
 		for(i = 0; i < n; i++) {
 			tmp = a[i*y+k];
 			for(j = 0; j < m; j++) {
 				// the offset is used to find out the relative position of the sub-matrix in the whole matrix
 				row_offset = (my_rank/n)*n;
 				col_offset = (my_rank%n)*n;
 				c[(i+row_offset)*y+(j+col_offset)] = c[(i+row_offset)*y+(j+col_offset)] + tmp * b[k*m+j];
 			}
 		}
 	}

 }


 void trans_matrix(double* a, double*b, int row_a, int col_a)
 {

 	int c,d;
 	for( c = 0 ; c < row_a ; c++ )
 	{
 		for( d = 0 ; d < col_a ; d++ )
 		{
 			b[d*row_a+c] = a[c*col_a+d];
 		}
 	}
 }


 int main(int argc, char *argv[])
 {
 	double* c;

 	double* a_row;
 	double* tmp_a_row;
 	double* b_col;
 	double* c_all;

 	MPI_Comm row_comm, col_comm, col_comm_1, col_comm_2, comm_world;

 	double start_time, end_time;
 	int world_rank,row_rank,col_rank, num_procs, n;

 	int myn;
 	int i,j,irow,jcol;

 	int my_start_row,my_start_col;
	/* Initialize MPI environment */ 
 	MPI_Init(&argc, &argv);
	/* Get rank of each MPI process */
 	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	/* Get overall number of processes */
 	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	{ /* Unnamed validation block */

 	if ( argc < 2 ) {
 		if (!world_rank)
 			printf("Usage: mpirun -n <num_procs> %s <n>\n", argv[0]);
 		MPI_Finalize();
 		exit(0);
 	}

		  /* Read n from command line argument */
 	n = atoi(argv[1]);

 	if ( n < num_procs ) {
 		if ( !world_rank )
 			printf("n should be greated than the number of processes\n");
 		MPI_Finalize();
 		exit(0);
 	}

 	if ( n % num_procs ) {
 		MPI_Finalize();
 		if ( !world_rank )
 			printf("n should divide the number of processes\n");
 		exit(0);  
 	}

	} /* End of the validation block */
 	myn = n/sqrt(num_procs);


 	irow = world_rank/myn;
 	jcol = world_rank%myn;
 	comm_world = MPI_COMM_WORLD;

 	MPI_Comm_split(comm_world, irow, jcol, &row_comm);
 	int size_row_comm;
 	MPI_Comm_size(row_comm,&size_row_comm);
 	MPI_Comm_split(comm_world, jcol, irow, &col_comm);

 	// get the rank in row and col communicator
 	MPI_Comm_rank(row_comm, &row_rank);
 	MPI_Comm_rank(col_comm, &col_rank);

 	// check rank
 	printf("     world_rank     irow     jcol   row_rank   col_rank\n");
 	printf("%8d %8d %8d %8d %8d\n",world_rank,irow,jcol,row_rank,col_rank);
 	MPI_Barrier(MPI_COMM_WORLD);

 	a_row =malloc(myn*n*sizeof(double));
 	b_col =malloc(n*myn*sizeof(double));
 	c = malloc(n*n*sizeof(double));
 	if(world_rank == 0)
 	{
 		c_all = malloc(n*n*sizeof(double));
 	}
 	MPI_Barrier(MPI_COMM_WORLD);

	  //init_matrix a
 	for(i=0; i<myn*myn; i++)
 	{
 		if(world_rank%myn == 1)
 			a_row[i+n] = world_rank;
 		else
 			a_row[i] = world_rank;


 		if(world_rank/myn == 1)
 			b_col[i+n] = world_rank*2;
 		else
 			b_col[i] = world_rank*2;
 	}

 	{
 		print_matrix(a_row,myn,n);
 		print_matrix(b_col,n,myn);
 	}

//printf("%f %f\n", buf[0],buf[1]);
 	MPI_Barrier(MPI_COMM_WORLD);
	//gather the whole row for a, whole colom for b
 


 	for(i=0;i<myn;i++)
 	{
 		MPI_Bcast(&a_row[i*n], n, MPI_DOUBLE, i, row_comm);
 		MPI_Bcast(&b_col[i*n], n, MPI_DOUBLE, i, col_comm);
 	}


 	MPI_Barrier(MPI_COMM_WORLD);


	//print_matrix(a,myn,myn);

 	{
 		printf("%s\n", "after boardcast");
 		tmp_a_row = malloc(n*myn*sizeof(double));
 		trans_matrix(a_row, tmp_a_row, n, myn);
 		a_row = tmp_a_row;

 		print_matrix(a_row,myn,n);
 		print_matrix(b_col,n,myn);
 	}

	//print_matrix(&a_ro[(world_rank/myn)*n],myn,n);
	//print_matrix(&b_col[(world_rank*myn)*n],n,myn);

 	MPI_Barrier(MPI_COMM_WORLD);
 	multiply_matrix(a_row, b_col, c, myn,n,myn,world_rank);

 	print_matrix(c,n,n);



 	MPI_Barrier(MPI_COMM_WORLD);

 	MPI_Reduce(c, c_all,n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 	if(world_rank ==0)
 	{
 		print_matrix(c_all,n,n);
 	}
 	

 	MPI_Finalize();
 	exit(0);

 }




