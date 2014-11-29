/*
 *  mmult.c: matrix multiplication using MPI.
 * There are some simplifications here. The main one is that matrices B and C 
 * are fully allocated everywhere, even though only a portion of them is 
 * used by each processor (except for processor 0)
 */

#include <mpi.h>
#include <stdio.h>
#include <math.h>

#include "io.c"

#define MPI_TIME_OUTPUT_FILENAME "out_mpi_time.csv"
#define MPI_RESULT_OUTPUT_FILENAME "out_mpi_result.csv"

 void write_data_to_file(char *fn, int n, double d)
 {
 	FILE *fp;
 	fp = fopen(fn,"a");
 	fprintf(fp, "%d,%lf\n", n, d);
 	fclose(fp);
 }


 void deleteOutputFile(char *fn)
 {
 	int status;
 	
 	status = remove(fn);
 	
 	if( status == 0 )
 		printf("%s file deleted successfully.\n",fn);
 	else
 	{
 		printf("Unable to delete the file\n");
 		perror("Error");
 	}
 }
 void print_matrix(double* m, int row, int col)
 {
 	int i, j = 0;
 	for (i=0; i<row; i++) {
 		printf("\n\t| ");
 		for (j=0; j<col; j++)
 			printf("%2d ", (int)m[i*col+j]);
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
 				row_offset = (my_rank/2)*n;
 				col_offset = (my_rank%2)*n;
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
 	double* a;
 	double* b;
 	double* c;

 	double* a_row;
 	double* tmp_a_row;
 	double* b_col;
 	double* c_all;
 	double start_time, end_time;
 	int myn;
 	int i,j,irow,jcol;

 	MPI_Comm row_comm, col_comm, comm_world;
 	int world_rank,row_rank,col_rank, num_procs, n;

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
 	int sqrt_num_procs = sqrt(num_procs);

 	irow = world_rank/sqrt_num_procs;
 	jcol = world_rank%sqrt_num_procs;
 	comm_world = MPI_COMM_WORLD;

 	MPI_Comm_split(comm_world, irow, jcol, &row_comm);
 	int size_row_comm;
 	MPI_Comm_size(row_comm,&size_row_comm);
 	MPI_Comm_split(comm_world, jcol, irow, &col_comm);

 	// get the rank in row and col communicator
 	MPI_Comm_rank(row_comm, &row_rank);
 	MPI_Comm_rank(col_comm, &col_rank);

 	// check rank
 	//printf("     world_rank     irow     jcol   row_rank   col_rank\n");
 	//printf("%8d %8d %8d %8d %8d\n",world_rank,irow,jcol,row_rank,col_rank);
 	MPI_Barrier(MPI_COMM_WORLD);

 	a = malloc(myn*myn*sizeof(double));
 	a_row =malloc(myn*n*sizeof(double));
 	b = malloc(myn*myn*sizeof(double));
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

 		if(world_rank%sqrt_num_procs == 1)
 			a_row[i+myn*myn] = 1.0;
 		else
 			a_row[i] = 1.0;

 		if(world_rank/sqrt_num_procs == 1)
 			b_col[i+myn*myn] = 2.0;
 		else
 			b_col[i] = 2.0;

 	}
 	/* Start time on process zero */
 	if (world_rank == 0)
 		start_time = MPI_Wtime(); 

 	
 	{
 		//print out the inital sub matrix
 		//print_matrix(a_row,myn,n);
 		//print_matrix(b_col,n,myn);
 	}

 	MPI_Barrier(MPI_COMM_WORLD);

 	for(i=0;i<sqrt_num_procs;i++)
 	{
 		MPI_Bcast(&a_row[i*myn*myn], myn*myn, MPI_DOUBLE, i, row_comm);
 		MPI_Bcast(&b_col[i*myn*myn], myn*myn, MPI_DOUBLE, i, col_comm);
 	}

 	MPI_Barrier(MPI_COMM_WORLD);

 	{
 		//printf("%s\n", "after boardcast");
 		tmp_a_row = malloc(n*myn*sizeof(double));
 		trans_matrix(a_row, tmp_a_row, n, myn);
 		a_row = tmp_a_row;
 		// the row and col for each process
 		//print_matrix(a_row,myn,n);
 		//print_matrix(b_col,n,myn);
 	}

	//print_matrix(&a_ro[(world_rank/myn)*n],myn,n);
	//print_matrix(&b_col[(world_rank*myn)*n],n,myn);
 	multiply_matrix(a_row, b_col, c, myn,n,myn,world_rank);

 	//print_matrix(c,n,n);

 	MPI_Barrier(MPI_COMM_WORLD);

	/* End time on process zero */
 	if (world_rank == 0)
 	{
 		end_time = MPI_Wtime();
 		printf("Multiplication takes %f sec\n", end_time- start_time);
 		write_data_to_file(MPI_TIME_OUTPUT_FILENAME, n, end_time- start_time);
 	}

 	MPI_Reduce(c, c_all,n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

 	MPI_Barrier(MPI_COMM_WORLD);
 	if(world_rank ==0)
 	{
		//print_matrix(c_all,n,n);
 		write_data_to_file(MPI_RESULT_OUTPUT_FILENAME,n, c_all[n*n-1]);
 		printf("%f\n", c_all[n*n-1]);
 	}

 	free(a_row);
 	free(b_col);
 	free(c);
 	if(world_rank==0)
 		free(c_all);

 	MPI_Finalize();
 	exit(0);
 }




