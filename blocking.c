#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"


struct timeval startwtime, endwtime1, endwtime2;
double seq_time1, seq_time2;

double **minDist;
double **minLabels;


double* packer(double **received, int blocksize, int LINESIZE);
double** unpacker(double *toReceive,int blocksize, int LINESIZE);
void knnSearch(int rank, int l, double **local, double **received, int blocksize,int LINESIZE, int nbrs);
void pointCompare(long i, long j, int nbrs, double *pointA, double *pointB, int LINESIZE);
double Ndistance(double *pointA, double *pointB, int LINESIZE);
void bubbleSort(int i,int nbrs);
void swap(int i, int k);

int main(int argc, char** argv){

	char filename[100];
	int MAX = 0;
	int LINESIZE = 0;
	int nbrs = 0;
	int i,j,l;
	int blocksize = 0;
	double **local;
	double **received;
	double *toSend;
	double *toReceive;
	int rank;
	int p; //==number of procs
	MPI_File fp;
	MPI_File fpResults;
	MPI_Status status;
	MPI_Offset offset;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(argc!=4){
		printf("ERROR: usage:\n%s filename MAX nbrs\n", argv[0]);
		printf("filename is the name of the .bin file to take data from\n");
		printf("MAX is the number of elements to take part in the search\n");
		printf("nbrs is the number of the nearest neighbours search for each point\n");
		exit(1);
	}

	MAX = atoi(argv[2]);
	nbrs = atoi(argv[3]);
	blocksize = MAX/p;
	
	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		LINESIZE = 30;
	}
	else{
		LINESIZE = 784;
	}
	
	//creating blocks
	local = (double **) malloc(blocksize*sizeof(double*));
	received = (double **) malloc(blocksize*sizeof(double*));
	toSend = (double *) malloc(blocksize*LINESIZE*sizeof(double));
	toReceive = (double *) malloc(blocksize*LINESIZE*sizeof(double));
	for(i=0; i<blocksize; i++){

		local[i] = (double *) malloc(LINESIZE*sizeof(double));
		received[i] = (double *) malloc(LINESIZE*sizeof(double));
	}
		
	//initialising results array
	minDist = (double **) malloc(MAX*sizeof(double*));
	minLabels = (double **) malloc(MAX*sizeof(double*));
	for(i=0; i<MAX; i++){
		minDist[i] = (double *) malloc(nbrs*sizeof(double));
		minLabels[i] = (double *) malloc(nbrs*sizeof(double));
		for(j=0; j<nbrs; j++){
			//presetting minDist to sth very big
			minDist[i][j] = 1000;
			minLabels[i][j] = -1;
		}
	}
	
	strcat(filename, ".bin");

	if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp)){
		printf("Error reading file from Process %d (fp)\n", rank);
		exit(1);
	}
	else{
		//block reading
		if(rank==0){
			printf("Initialising kNN search for problem size %d and k = %d\nusing archive %s\n", MAX, nbrs, filename);
		}

		for(i=0; i<blocksize; i++){

			for(j=0; j<LINESIZE ; j++){
				offset = rank*blocksize*LINESIZE*sizeof(double)+i*LINESIZE*sizeof(double)+j*sizeof(double);
				MPI_File_read_at(fp, offset, &local[i][j], 1, MPI_DOUBLE, &status);
			}
		}
		//blockprint(local, blocksize, rank);
		
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_File_close(&fp);
		knnSearch(rank, rank, local, local, blocksize,LINESIZE, nbrs);
	}
	
	//2d array to 1d array:
	toSend = packer(local, blocksize, LINESIZE);
	
	if(rank==0) gettimeofday (&startwtime, NULL);

	//circulation of blocks
	for(l=0; l<(p-1); l++){

		if(rank!=0){
			MPI_Recv(toReceive, blocksize*LINESIZE, MPI_DOUBLE, rank-1, 10, MPI_COMM_WORLD, &status);
		}
		
		if(rank!=(p-1)){
			MPI_Ssend(toSend, blocksize*LINESIZE, MPI_DOUBLE, rank+1, 10, MPI_COMM_WORLD);
		}
		else{
			MPI_Ssend(toSend, blocksize*LINESIZE, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
		}

		if(rank==0){
			MPI_Recv(toReceive, blocksize*LINESIZE, MPI_DOUBLE, p-1, 10, MPI_COMM_WORLD, &status);
		}

		//1d to 2d array
		received = unpacker(toReceive, blocksize, LINESIZE);
		
		int tmp = status.MPI_SOURCE;
		for(int t=0; t<l ;t++){
			tmp--;
			if(tmp<0) tmp = p-1;
		}

		//time for blocking COMMS
		//only on the last rep will count
		//rank 0 recieves last
		if((rank==0) && (l==(p-2)) ) gettimeofday (&endwtime1, NULL);
		
		knnSearch(rank, tmp, local, received, blocksize, LINESIZE, nbrs);

		toSend = packer(received, blocksize, LINESIZE);
	}


	//preparing to send results to proc 0
	toSend = (double *) realloc(toSend, blocksize*nbrs*sizeof(double));
	toReceive = (double *) realloc(toReceive, blocksize*nbrs*sizeof(double));
	
	if(rank==0){
		for(i=1; i<p; i++){

			MPI_Recv(toSend, blocksize*nbrs, MPI_DOUBLE, i, 15, MPI_COMM_WORLD, &status);

			received = unpacker(toSend, blocksize, nbrs);

			for(j=0; j<blocksize; j++){
				for(int k=0; k<nbrs; k++){
					minDist[status.MPI_SOURCE*blocksize+j][k] = received[j][k];
				}
			}

			MPI_Recv(toReceive, blocksize*nbrs, MPI_DOUBLE, i, 20, MPI_COMM_WORLD, &status);

			received = unpacker(toReceive, blocksize, nbrs);

			for(j=0; j<blocksize; j++){
				for(int k=0; k<nbrs; k++){
					minLabels[status.MPI_SOURCE*blocksize+j][k] = received[j][k];
				}
			}
		}
		
	}
	else{
		//toSend buffer used for minDist
		//toReceive buffer used for minLabels
		for(i=0; i<blocksize; i++){
			for(j=0; j<nbrs; j++){
				toSend[i*nbrs+j] = minDist[rank*blocksize+i][j];
				toReceive[i*nbrs+j] = minLabels[rank*blocksize+i][j];
			}
		}

		MPI_Send(toSend, blocksize*nbrs, MPI_DOUBLE, 0, 15, MPI_COMM_WORLD);
		MPI_Send(toReceive, blocksize*nbrs, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD);
	}

	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		strcpy(filename, "results-mpi-blocking-svd.txt");
	}
	else{
		strcpy(filename, "results-mpi-blocking.txt");
	}

	if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp)){
		printf("Error opening file from Process %d (fp)\n", rank);
		exit(1);
	}




	if(rank==0){
		gettimeofday (&endwtime2, NULL);
	}
	
/*
	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		strcpy(filename, "results-mpi-blocking-labels-svd.txt");
	}
	else{
		strcpy(filename, "results-mpi-blocking-labels.txt");
	}

	if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp)){
		printf("Error opening file from Process %d (fp)\n", rank);
		exit(1);
	}

	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		strcpy(filename, "results-mpi-blocking-dist-svd.txt");
	}
	else{
		strcpy(filename, "results-mpi-blocking-dist.txt");
	}

	if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fpResults)){
		printf("Error opening file from Process %d (fp)\n", rank);
		exit(1);
	}

	//printing results in two seperate files
	for(i=0; i<blocksize; i++){
		char buf[100];
		for(j=0; j<nbrs; j++){
			if(minLabels[rank*blocksize+i][j]==-1) printf("ERROR\n");
			//offset = rank*blocksize*LINESIZE*sizeof(char)+i*LINESIZE*sizeof(char)+j*sizeof(char);
			offset = rank*blocksize*LINESIZE*sizeof(char)+i*LINESIZE*sizeof(char)+j*sizeof(char);
			sprintf(buf, "%f ", minLabels[rank*blocksize+i][j]);
			MPI_File_write_at(fp, offset, buf, strlen(buf), MPI_CHAR, &status);
			sprintf(buf, "%f ", minDist[rank*blocksize+i][j]);
			MPI_File_write_at(fpResults, offset, buf, strlen(buf), MPI_CHAR, &status);
			//printf("#%d: %d with a distance of %f\n", j+1, (int) minLabels[i][j], minDist[i][j]);
		}
	}
	*/
	
	if(rank==0){
		//printing results in a sigle file in easily readable form from proc 0 ONLY
		for(i=0; i<p*blocksize; i++){
			char buf[100];
			sprintf( buf, "Top %d closest to point %d:\n",nbrs, i);
			MPI_File_write(fp, buf, strlen(buf), MPI_CHAR, &status);
			//printf("Top %d closest to point %d:\n",nbrs, i);
			for(j=0; j<nbrs; j++){
				if(minLabels[i][j]==-1) printf("ERROR\n");
				sprintf(buf, "#%d: %d with a distance of %f\n", j+1, (int) minLabels[i][j], minDist[i][j]);
				MPI_File_write(fp, buf, strlen(buf), MPI_CHAR, &status);
				//printf("#%d: %d with a distance of %f\n", j+1, (int) minLabels[i][j], minDist[i][j]);
			}
		}
	}

	if(rank==0){
		seq_time1 = (double)((endwtime1.tv_usec - startwtime.tv_usec)/1.0e6
			      + endwtime1.tv_sec - startwtime.tv_sec);

		printf("COMMS Wall clock time = %f\n", seq_time1);

		seq_time2 = (double)((endwtime2.tv_usec - startwtime.tv_usec)/1.0e6
			      + endwtime2.tv_sec - startwtime.tv_sec);

		printf("FINAL Wall clock time = %f\n", seq_time2);

		printf("\nJob Done.\n");
	}

	MPI_File_close(&fp);
	MPI_File_close(&fpResults);

	free(local);
	free(received);
	free(toSend);
	free(toReceive);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return (0);
}

double* packer(double **received,int blocksize, int LINESIZE){
	int i,j;
	double *temp;
	temp = (double *) malloc(blocksize*LINESIZE*sizeof(double));
	for(i=0; i<blocksize; i++){
		for(j=0; j<LINESIZE; j++){
			temp[i*LINESIZE+j]=received[i][j];
		}
	}
	return temp;

}

double** unpacker(double *toReceive,int blocksize, int LINESIZE){
	int i,j;
	double **temp;
	temp = (double **) malloc(blocksize*sizeof(double));
	for(i=0; i<blocksize; i++){
		temp[i] = (double *) malloc(LINESIZE*sizeof(double));
		for(j=0; j<LINESIZE; j++){
			temp[i][j]= toReceive[i*LINESIZE+j];
		}
	}
	return temp;

}


void knnSearch(int rank, int l, double **local, double **received, int blocksize, int LINESIZE, int nbrs){
	int i,j,k;
	double *pointA;
	double *pointB;
	
	pointA = (double *) malloc(LINESIZE*sizeof(double));
	pointB = (double *) malloc(LINESIZE*sizeof(double));

	for(i=0; i<blocksize; i++){

		//reading pointA from block local
		for(k=0; k<LINESIZE; k++){
			pointA[k]=local[i][k];
		}

		for(j=0; j<blocksize; j++){

			//reading pointB from block received
			for(k=0; k<LINESIZE; k++){
				pointB[k]=received[j][k];
			}
			pointCompare(rank*blocksize+i, l*blocksize+j, nbrs, pointA, pointB, LINESIZE);
		}
	}
}

void pointCompare(long i, long j, int nbrs, double *pointA, double *pointB, int LINESIZE){

	double dist=0;
	int k,n;

	//calculating distance
	dist=Ndistance(pointA, pointB, LINESIZE);
	//sorting top k closest neighbours
	bubbleSort(i, nbrs);
	for(n=0; n<nbrs ; n++){
		//if dist = 0 then pointA=pointB
		if(dist>0 && dist<minDist[i][n]){
			//pushing back all elements 
			//from the end to the point where this new dist will be inserted
			for(k=(nbrs-1); k>n; k--){
				minDist[i][k] = minDist[i][k-1];
				minLabels[i][k] = minLabels[i][k-1];
			}
			minDist[i][n] = dist;
			minLabels[i][n] = j;
			break;
		}
	}
	
	
}

double Ndistance(double *pointA, double *pointB, int LINESIZE){
	double dist=0;
	for(int k=0; k<LINESIZE; k++){
		dist += pow(pointA[k]-pointB[k],2);
	}
	return sqrt(dist);
}

void bubbleSort(int i,int nbrs){
	int j,k;
	for(j=0; j<nbrs; j++){
		for(k=(nbrs-1); k>j; k--){
			if(minDist[i][k-1]>minDist[i][k]){
				swap(i,k);
			}
		}
	}
}

void swap(int i, int k){
	double tmp;
	tmp = minDist[i][k-1];
	minDist[i][k-1] = minDist[i][k];
	minDist[i][k] = tmp;
	tmp = minLabels[i][k-1];
	minLabels[i][k-1] = minLabels[i][k];
	minLabels[i][k] = tmp;
}