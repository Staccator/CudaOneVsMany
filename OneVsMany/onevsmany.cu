#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "choice_methods.h"
#include "read_write.h"

using namespace std;

void StartTimer(StopWatchInterface *timer)
{
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
}
void StopTimer(StopWatchInterface *timer)
{
	sdkStopTimer(&timer);
	printf("[GPU] processing time : %f (ms)\n", sdkGetTimerValue(&timer));
}

void CalculateUserIndexes(int userCount, int* hostUserRows, int* &deviceUserRows, int* &hostBasicIndexes, int* &deviceBasicIndexes, int* &hostUserIndexes, int* &deviceUserIndexes, float* &hostDistancesUser, float* &deviceDistancesUser, int &totalSelfUsersMemory)
{
	int userIndexesMemory = userCount * sizeof(int);
	hostBasicIndexes = (int*)malloc(userIndexesMemory);
	hostUserIndexes = (int*)malloc(userIndexesMemory);
	int lastIndex = 0;
	int lastBasicIndex = 0;
	for (int i = 0; i < userCount; i++)
	{
		hostUserIndexes[i] = lastIndex;
		hostBasicIndexes[i] = lastBasicIndex;
		int rows = hostUserRows[i];
		int memory = rows * rows;
		lastIndex += memory;
		lastBasicIndex += rows;
	}
	totalSelfUsersMemory = lastIndex * sizeof(float);
	hostDistancesUser = (float*)malloc(totalSelfUsersMemory);

	checkCudaErrors(cudaMalloc((void **)&deviceUserRows, userIndexesMemory));
	checkCudaErrors(cudaMemcpy(deviceUserRows, hostUserRows, userIndexesMemory, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&deviceUserIndexes, userIndexesMemory));
	checkCudaErrors(cudaMemcpy(deviceUserIndexes, hostUserIndexes, userIndexesMemory, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&deviceBasicIndexes, userIndexesMemory));
	checkCudaErrors(cudaMemcpy(deviceBasicIndexes, hostBasicIndexes, userIndexesMemory, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&deviceDistancesUser, totalSelfUsersMemory));
	if (!hostDistancesUser || !deviceDistancesUser)
		printf("Memory allocation for user failed\n");
}

__host__ __device__ inline static
float Distance(float* dataTab, int columnCount, int x, int y)
{
	int column = 1;
	float xVal = dataTab[x * columnCount + column];
	float yVal = dataTab[y * columnCount + column];
	return abs(xVal - yVal);
}

__global__ void
DistancesPerUser(int* userRows, int* resultUserIndexes, int* basicUserIndexes, int columnCount, float* dataTab, float* distancesUser)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int blockSize = blockDim.x;
	int rows = userRows[blockId];
	int cellsToCalculate = rows * rows;
	int resultStartIndex = resultUserIndexes[blockId];
	int basicStartIndex = basicUserIndexes[blockId];
	//printf("Block=%d, Tid=%d, Rows=%d, StartIndex=%d\n", blockId, threadId, rows, startIndex);
	for (int i = threadId; i < cellsToCalculate; i += blockSize)
	{
		int x = basicStartIndex + i % rows;
		int y = basicStartIndex + i / rows;
		distancesUser[resultStartIndex + i] = Distance(dataTab, columnCount, x, y);
	}
}

__global__ void
oneVsManyRowWise(float* dataTab, int rowCount, int columnCount, float* resultTab, int* usersRowsIndexes)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= rowCount) return;
	int myIndex = usersRowsIndexes[tid];
	int resultShift = rowCount * tid;

	for (int i = 0; i < rowCount; i++)
	{
		int otherIndex = usersRowsIndexes[i];
		resultTab[resultShift + i] = Distance(dataTab, columnCount, myIndex, otherIndex);
	}
}

__global__ void
oneVsManyUserBlock(float* dataTab, int rowCount, int columnCount, float* resultTab, int* usersRowsIndexes, int* userStartIndexes, int userCount)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int blockSize = blockDim.x;
	int userRows;
	if (blockId == userCount - 1)
		userRows = rowCount - userStartIndexes[userCount - 1];
	else 
		userRows = userStartIndexes[blockId + 1] - userStartIndexes[blockId];

	int cellsToCalculate = userRows * rowCount;
	int userStartIndex = userStartIndexes[blockId];
	int resultShift = userStartIndex * rowCount;
	//printf("Block=%d, Tid=%d, Rows=%d, Shift=%d\n", blockId, threadId, userRows, resultShift);
	for (int i = threadId; i < cellsToCalculate; i += blockSize)
	{
		int x = usersRowsIndexes[(resultShift + i) % rowCount];
		int y = usersRowsIndexes[(resultShift + i) / rowCount];
		resultTab[resultShift + i] = Distance(dataTab, columnCount, x, y);
	}
}


int main(int argc, char **argv)
{
	// Config
	auto config = ReadConfig();
	int THREADS_PER_BLOCK = stoi(config["THREADS_PER_BLOCK"]);
	int ROW_SELECTION_METHOD = stoi(config["ROW_SELECTION_METHOD"]);
	int ONE_VS_MANY_METHOD = stoi(config["ONE_VS_MANY_METHOD"]);
	LoadChoiceMethodsConfigValues();

	// Main
    srand(7312);
	int tpb = THREADS_PER_BLOCK;
    StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);

	float* hostDataTab;
    float * deviceDataTab;
	int* hostUserTab;
	int* deviceUserTab;	
	int rowCount;
	int columnCount;

	int* hostUserRows;
	int* deviceUserRows;
	int userCount;
	float* hostDistancesUser;
	float* deviceDistancesUser;
	int* hostUserResultIndexes;
	int* deviceUserResultIndexes;
	int* hostUserBasicIndexes;
	int* deviceUserBasicIndexes;
	int totalSelfUsersMemory;
	vector<string> ids;
	int* sessionNumbers;

	ReadData(rowCount, columnCount, userCount, hostDataTab, hostUserTab, hostUserRows, ids, sessionNumbers);

	// Memory allocation
	int dataMemory = sizeof(float) * rowCount * columnCount;
	int userMemory = sizeof(int) * rowCount;
    checkCudaErrors(cudaMalloc((void **)& deviceDataTab, dataMemory));
    checkCudaErrors(cudaMemcpy(deviceDataTab, hostDataTab, dataMemory, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)& deviceUserTab, userMemory));
	checkCudaErrors(cudaMemcpy(deviceUserTab, hostUserTab, userMemory, cudaMemcpyHostToDevice));
	
	if(!hostDataTab || !deviceDataTab || !hostUserTab || !deviceUserTab) printf("Memory allocation failed\n");

	//Distances Per User
	CalculateUserIndexes(userCount, hostUserRows, deviceUserRows, hostUserBasicIndexes, deviceUserBasicIndexes, hostUserResultIndexes, deviceUserResultIndexes, hostDistancesUser, deviceDistancesUser, totalSelfUsersMemory);
	//PrintTab(hostUserRows, userCount);
	//PrintTab(hostUserResultIndexes, userCount);

	//StartTimer(timer);
	DistancesPerUser << < userCount, tpb >> > (deviceUserRows, deviceUserResultIndexes, deviceUserBasicIndexes, columnCount, deviceDataTab, deviceDistancesUser);
	cudaDeviceSynchronize();
	//StopTimer(timer);
	checkCudaErrors(cudaMemcpy(hostDistancesUser, deviceDistancesUser, totalSelfUsersMemory, cudaMemcpyDeviceToHost));
	//PrintUsersSelfDistances(userCount, hostUserRows, hostUserResultIndexes, hostDistancesUser);

	//You need to specify below values for usage in oneVsMany
	int sumOfSelectedRows;
	int * usersRowsIndexes;
	int * userStartIndexes = (int*)malloc(userCount * sizeof(int));
	
	//Selecting rows from groups
	switch (ROW_SELECTION_METHOD)
	{
		case 1:
			sumOfSelectedRows = rowCount;
			usersRowsIndexes = ChooseAll(rowCount);
			userStartIndexes = hostUserBasicIndexes;
			break;
		case 2:
			sumOfSelectedRows = ChooseRandom(usersRowsIndexes, userStartIndexes, hostUserRows, hostUserBasicIndexes, userCount);
			break;
		case 3:
			sumOfSelectedRows = ChooseHeuristic(usersRowsIndexes, userStartIndexes, userCount, hostUserRows, hostUserResultIndexes, hostDistancesUser, hostUserBasicIndexes);
			break;
		case 4:
			sumOfSelectedRows = ChooseCharacteristicAreas(usersRowsIndexes, userStartIndexes, userCount, hostUserRows, hostUserResultIndexes, hostDistancesUser, hostUserBasicIndexes);
			break;
		case 5:
			sumOfSelectedRows = ChooseWithoutDistant(usersRowsIndexes, userStartIndexes, userCount, hostUserRows, hostUserResultIndexes, hostDistancesUser, hostUserBasicIndexes);
			break;
	}
	//PrintSelectedRows(sumOfSelectedRows, userCount, usersRowsIndexes, userStartIndexes);
	
	//Memory Allocation for OneVsMany
	int resultMemory = sizeof(float) * sumOfSelectedRows * sumOfSelectedRows;
	float * hostResultTab = (float*)malloc(resultMemory);
	float * deviceResultTab;
	checkCudaErrors(cudaMalloc((void **)&deviceResultTab, resultMemory));
	int * deviceUsersRowsIndexes;
	int usersRowsIndexesMemory = sumOfSelectedRows * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&deviceUsersRowsIndexes, usersRowsIndexesMemory));
	checkCudaErrors(cudaMemcpy(deviceUsersRowsIndexes, usersRowsIndexes, usersRowsIndexesMemory, cudaMemcpyHostToDevice));

	int * deviceUserStartIndexes;
	int userStartIndexesMemory = userCount * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&deviceUserStartIndexes, userStartIndexesMemory));
	checkCudaErrors(cudaMemcpy(deviceUserStartIndexes, userStartIndexes, userStartIndexesMemory, cudaMemcpyHostToDevice));
	if (!hostResultTab || !deviceResultTab || !deviceUsersRowsIndexes || !deviceUserStartIndexes) printf("Memory allocation failed\n");

	// OneVsMany Calculations
	int numberOfBlocks;
    StartTimer(timer);
	switch (ONE_VS_MANY_METHOD) {
	case 1:
		numberOfBlocks = sumOfSelectedRows / tpb + 1;
		oneVsManyRowWise << < numberOfBlocks, tpb >> > (deviceDataTab, sumOfSelectedRows, columnCount, deviceResultTab, deviceUsersRowsIndexes);
		break;
	case 2:
		numberOfBlocks = userCount;
		oneVsManyUserBlock << < numberOfBlocks, tpb >> > (deviceDataTab, sumOfSelectedRows, columnCount, deviceResultTab, deviceUsersRowsIndexes, deviceUserStartIndexes, userCount);
		break;
	}
	cudaDeviceSynchronize();
	StopTimer(timer);
	checkCudaErrors(cudaMemcpy(hostResultTab, deviceResultTab, resultMemory, cudaMemcpyDeviceToHost));

	//Print Results
	//PrintTab2D(sumOfSelectedRows, hostResultTab);
	WriteResults(hostResultTab, sumOfSelectedRows, userCount, usersRowsIndexes, userStartIndexes, ids, sessionNumbers);

	//Free memory and timer
	sdkDeleteTimer(&timer);
	free(hostDataTab); free(hostUserTab); free(usersRowsIndexes); free(hostDistancesUser);free(hostUserBasicIndexes); free(hostUserResultIndexes);
	cudaFree(deviceDataTab); cudaFree(deviceUserTab); cudaFree(deviceUsersRowsIndexes); cudaFree(deviceDistancesUser);cudaFree(deviceUserBasicIndexes); cudaFree(deviceUserResultIndexes);
    getLastCudaError("Kernel execution failed");
}
