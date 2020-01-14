#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

struct timeval startwtime, endwtime;
double seq_time;


double **minDist;
double **minLabels;

void knnSearch(int outerchunk, int innercunk, double **prev, double **next, int blocksize,int LINESIZE, int nbrs);
void pointCompare(long i, long j, int nbrs, double *pointA, double *pointB, int LINESIZE);
double Ndistance(double *pointA, double *pointB, int LINESIZE);
void bubbleSort(int i,int nbrs);
void swap(int i, int k);

int main(int argc, char** argv){

	FILE *fp = NULL;
	FILE *fpResults = NULL;
	char filename[100];
	int MAX = 0;
	int LINESIZE = 0;
	int nbrs = 0;
	int i,j,k,l;
	int chunk = 0;
	int blocksize = 0;
	double** prev;
	double** next;
	
	if(argc!=5){
		printf("ERROR: usage:\n%s filename MAX LINESIZE chunk nbrs\n", argv[0]);
		printf("filename is the name of the .bin file to take data from\n");
		printf("MAX is the number of elements to take part in the search\n");
		printf("chunk is the number of chunks the problem will be divided in\n");
		printf("nbrs is the number of the nearest neighbours search for each point\n");
		exit(1);
	}

	MAX = atoi(argv[2]);
	chunk = atoi(argv[3]);
	nbrs = atoi(argv[4]);
	blocksize = MAX/chunk;
	
	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		LINESIZE = 30;
	}
	else{
		LINESIZE = 784;
	}
	
	//creating blocks
	prev = (double **) malloc(blocksize*sizeof(double*));
	next = (double **) malloc(blocksize*sizeof(double*));
	for(i=0; i<blocksize; i++){
		prev[i]=(double *) malloc(LINESIZE*sizeof(double));
		next[i]=(double *) malloc(LINESIZE*sizeof(double));
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
	//open file with fp ptr for pointA
	fp = fopen( filename ,"rb");
		if(fp == NULL){
			printf("Error reading file(fp)\n");
			exit(1);
		}


	printf("Initialising kNN search for problem size %d and k = %d\n", MAX, nbrs);
	printf("\n...reading data...");
	printf("- using archive %s\n", filename);

	gettimeofday (&startwtime, NULL);

	//i refers to No. of Block
	for(i=0; i<chunk; i++){
		fseek(fp, i*blocksize*LINESIZE*sizeof(double), SEEK_SET);
		//reading block prev
		for(j=0; j<blocksize && feof(fp)==0; j++){

			for(k=0; k<LINESIZE ; k++){
				if(fread(&prev[j][k], sizeof(double), 1, fp) != 1){
					printf("Error reading coordinates from file(fp)");
					exit(1);
				}
			}
		}
		//EDW EXW 1 BLOCK
		for(l=0; l<chunk; l++){
			fseek(fp, l*blocksize*LINESIZE*sizeof(double), SEEK_SET);
			for(j=0; j<blocksize && feof(fp)==0; j++){
				
				for(k=0; k<LINESIZE ; k++){
					if(fread(&next[j][k], sizeof(double), 1, fp) != 1){
						printf("Error reading coordinates from file(fp)");
						exit(1);
					}
				}
			}
			knnSearch(i, l, prev, next, blocksize,LINESIZE, nbrs);
		}
	}
	gettimeofday (&endwtime, NULL);
	fclose(fp);

	//printing results in a single file in easily readable form
	/*for(i=0;i<MAX;i++){
		fprintf(fpResults, "%s %d %s %d %s", "Top ", nbrs, "closest to point ", i, ":\n");
		for(j=0; j<nbrs; j++){
			if(minDist[i][j][0]==-1) printf("ERROR\n");
			fprintf(fpResults, "%s %d %s %d %s %f %s", "#", j+1, ": ", (int) minDist[i][j][0]," with a distance of ", minDist[i][j][1], "\n");
		}

	}*/
	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		strcpy(filename, "results_labels_svd.txt");
	}
	else{
		strcpy(filename, "results_labels.txt");
	}
	//for the labels
	fp = fopen(filename ,"w");
		if(fp == NULL){
			printf("Error reading file(fp)\n");
			exit(1);
		}

	strcpy(filename, argv[1]);
	if(!strcmp(filename,"trainX_svd")){
		strcpy(filename, "results_dist_svd.txt");
	}
	else{
		strcpy(filename, "results_dist.txt");
	}
	//for the distances
	fpResults = fopen(filename ,"w");
		if(fpResults == NULL){
		printf("Error reading file(fpB)\n");
		exit(1);
	}


	for(i=0;i<MAX;i++){
		for(j=0; j<nbrs; j++){
			if(minLabels[i][j]==-1) printf("ERROR\n");
			fprintf(fp, "%d ",(int) minLabels[i][j]);
			fprintf(fpResults, "%f ", minDist[i][j]);
		}
		fprintf(fp, "%s", "\n");
		fprintf(fpResults, "%s", "\n");
	}


	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

	printf("Wall clock time = %f\n", seq_time);

	fclose(fp);
	fclose(fpResults);

	printf("\nJob Done.\n");

	return (0);
}


void knnSearch(int outerchunk, int innercunk, double **prev, double **next, int blocksize,int LINESIZE, int nbrs){
	int i,j,k;
	double *pointA;
	double *pointB;

	pointA = (double *) malloc(LINESIZE*sizeof(double));
	pointB = (double *) malloc(LINESIZE*sizeof(double));

	for(i=0; i<blocksize; i++){

		//reading pointA from block local
		for(k=0; k<LINESIZE; k++){
			pointA[k]=prev[i][k];
		}

		for(j=0; j<blocksize; j++){

			//reading pointB from block received
			for(k=0; k<LINESIZE; k++){
				pointB[k]=next[j][k];
			}
			pointCompare(outerchunk*blocksize+i, innercunk*blocksize+j, nbrs, pointA, pointB, LINESIZE);
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
	minDist[i][k-1]=minDist[i][k];
	minDist[i][k]=tmp;
	tmp = minLabels[i][k-1];
	minLabels[i][k-1]=minLabels[i][k];
	minLabels[i][k]=tmp;
}