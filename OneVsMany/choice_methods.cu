#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <random>
#include <string>
#include <iterator>
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>

#include "choice_methods.h"
#include "read_write.h"

int MAX_ROWS_PER_USER;
// Method 4 & 5
int BUCKET_COUNT;
float BUCKET_SIZE;
// Method 4
int CHARACTERISTIC_BUCKETS_COUNT;
// Method 5
float BUCKET_SIZE_TO_BE_REJECTED;

using namespace std;

void LoadChoiceMethodsConfigValues()
{
	auto config = ReadConfig();

	MAX_ROWS_PER_USER = stoi(config["MAX_ROWS_PER_USER"]);
	BUCKET_COUNT = stoi(config["BUCKET_COUNT"]);
	BUCKET_SIZE = stof(config["BUCKET_SIZE"]);
	CHARACTERISTIC_BUCKETS_COUNT = stoi(config["CHARACTERISTIC_BUCKETS_COUNT"]);
	BUCKET_SIZE_TO_BE_REJECTED = stof(config["BUCKET_SIZE_TO_BE_REJECTED"]);
}

void PrintSelectedRows(int checkedRowsCount, int userCount, int* usersRowsIndexes, int* userStartIndexes)
{
	cout << "Selected Rows" << endl;
	for (int i = 0; i < userCount; i++)
	{
		cout << userStartIndexes[i] << " ";
	} cout << endl;

	int j = 0;
	for (int i = 0; i < checkedRowsCount; i++)
	{
		if (i == userStartIndexes[j])
		{
			cout << endl;
			j++;
		}
		cout << usersRowsIndexes[i] << " ";
	} cout << endl;
}

int* ChooseAll(int rowCount)
{
	int* result = (int*)malloc(rowCount * sizeof(int));
	for (int i = 0; i < rowCount; i++)
	{
		result[i] = i;
	}
	return result;
}

int ChooseRandom(int* &usersRowsIndexes, int* &userStartIndexes, int* hostUserRows, int* hostUserBasicIndexes, int userCount)
{
	int usersRowsIndexesMemory = 0;
	int rowsPerUser = MAX_ROWS_PER_USER;
	int lastIndex = 0;
	for (int i = 0; i < userCount; i++)
	{
		int selectedRowsCount = min(rowsPerUser, hostUserRows[i]);
		usersRowsIndexesMemory += selectedRowsCount;
		userStartIndexes[i] = lastIndex;
		lastIndex += selectedRowsCount;
	}

	usersRowsIndexes = (int*)malloc(usersRowsIndexesMemory * sizeof(int));
	for (int i = 0; i < userCount; i++)
	{
		int selectedRowsCount = min(rowsPerUser, hostUserRows[i]);

		std::vector<int> v;
		for (int j = 0; j < hostUserRows[i]; j++)
		{
			v.push_back(hostUserBasicIndexes[i] + j);
		}
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(v.begin(), v.end(), g);
		std::copy(v.begin(), v.begin() + selectedRowsCount, usersRowsIndexes + userStartIndexes[i]);
	}

	return usersRowsIndexesMemory;
}

int ChooseHeuristic(int* &usersRowsIndexes, int* &userStartIndexes, int userCount, int* userRows, int* startIndexes, float* distancesUser, int* hostUserBasicIndexes)
{
	int usersRowsIndexesMemory = 0;
	int rowsPerUser = MAX_ROWS_PER_USER;
	int lastIndex = 0;
	for (int i = 0; i < userCount; i++)
	{
		int selectedRowsCount = min(rowsPerUser, userRows[i]);
		usersRowsIndexesMemory += selectedRowsCount;
		userStartIndexes[i] = lastIndex;
		lastIndex += selectedRowsCount;
	}

	usersRowsIndexes = (int*)malloc(usersRowsIndexesMemory * sizeof(int));
	for (int i = 0; i < userCount; i++)
	{
		int rows = userRows[i];
		int startIndex = startIndexes[i];
		std::vector<float> sort_values;
		std::vector<int> v;
		for (int x = 0; x < rows; x++)
		{
			float sum = 0;
			for (int y = 0; y < rows; y++)
			{
				sum += distancesUser[startIndex + x * rows + y];
			}
			sort_values.push_back(sum);
			v.push_back(hostUserBasicIndexes[i] + x);
		}

		std::vector<std::size_t> index(v.size());
		std::iota(index.begin(), index.end(), 0);
		std::sort(index.begin(), index.end(), [&](size_t a, size_t b) { return sort_values[a] < sort_values[b]; });
		int selectedRowsCount = min(rowsPerUser, userRows[i]);
		for (int z = 0; z < selectedRowsCount; z++)
		{
			usersRowsIndexes[userStartIndexes[i] + z] = v[index[z]];
		}
	}

	return usersRowsIndexesMemory;
}

int ChooseWithoutDistant(int* &usersRowsIndexes, int* &userStartIndexes, int userCount, int* userRows, int* startIndexes, float* distancesUser, int* hostUserBasicIndexes)
{
	int bucketCount = BUCKET_COUNT;
	float bucketSize = BUCKET_SIZE;
	int* buckets = (int*)malloc(bucketCount * sizeof(int));

	int lastIndex = 0;
	vector<int> resultIndexes;

	for (int i = 0; i < userCount; i++)
	{
		int rows = userRows[i];
		int startIndex = startIndexes[i];
		for (int j = 0; j < bucketCount; j++)
		{
			buckets[j] = 0;
		}

		for (int x = 0; x < rows; x++)
		{
			for (int y = 0; y < rows; y++)
			{
				if (x == y) continue;
				float distance = distancesUser[startIndex + x * rows + y];
				int index = distance / bucketSize;
				if (index < bucketCount) 
					{ buckets[index] += 1; }
			}
		}

		int selectedRowsCount = 0;
		for (int x = 0; x < rows; x++)
		{
			bool isCorrect = true;
			for (int y = 0; y < rows; y++)
			{
				if (x == y) continue;
				float distance = distancesUser[startIndex + x * rows + y];
				int index = distance / bucketSize;
				if (index < bucketCount)
				{
					if (buckets[index] <= BUCKET_SIZE_TO_BE_REJECTED) {
						isCorrect = false; break;
					}
				}
				else
				{
					isCorrect = false; break;
				}
			}
			if (isCorrect) 
			{
				if (selectedRowsCount >= MAX_ROWS_PER_USER) continue;

				resultIndexes.push_back(hostUserBasicIndexes[i] + x);
				selectedRowsCount += 1;
			}
		}

		userStartIndexes[i] = lastIndex;
		lastIndex += selectedRowsCount;
	}

	usersRowsIndexes = (int*)malloc(resultIndexes.size() * sizeof(int));
	copy(resultIndexes.begin(), resultIndexes.end(), usersRowsIndexes);
	return resultIndexes.size();
}

int ChooseCharacteristicAreas(int* &usersRowsIndexes, int* &userStartIndexes, int userCount, int* userRows, int* startIndexes, float* distancesUser, int* hostUserBasicIndexes)
{
	int bucketCount = BUCKET_COUNT;
	float bucketSize = BUCKET_SIZE;
	int* buckets = (int*)malloc(bucketCount * sizeof(int));

	int lastIndex = 0;
	vector<int> resultIndexes;

	for (int i = 0; i < userCount; i++)
	{
		int rows = userRows[i];
		int startIndex = startIndexes[i];
		for (int j = 0; j < bucketCount; j++) buckets[j] = 0;

		for (int x = 0; x < rows; x++)
		{
			for (int y = 0; y < rows; y++)
			{
				if (x == y) continue;
				float distance = distancesUser[startIndex + x * rows + y];
				int index = distance / bucketSize;
				if (index < bucketCount)
				{
					buckets[index] += 1;
				}
			}
		}

		int selectedRowsCount = 0;
		std::vector<std::size_t> index(bucketCount);
		std::iota(index.begin(), index.end(), 0);
		std::sort(index.begin(), index.end(), [&](size_t a, size_t b) { return buckets[a] > buckets[b]; });
		for (int j = 0; j < bucketCount; j++) buckets[j] = 0;
		for (int j = 0; j < CHARACTERISTIC_BUCKETS_COUNT; j++) {
			buckets[index[j]] = 1;
		}
		//for (int j = 0; j < bucketCount; j++) cout << buckets[j] << " "; cout << endl;

		for (int x = 0; x < rows; x++)
		{
			bool isCorrect = false;
			for (int y = 0; y < rows; y++)
			{
				if (x == y) continue;
				float distance = distancesUser[startIndex + x * rows + y];
				int index = distance / bucketSize;
				if (index < bucketCount)
				{
					if (buckets[index] == 1)
					{
						isCorrect = true; break;
					}
				}
			}
			if (isCorrect)
			{
				if (selectedRowsCount >= MAX_ROWS_PER_USER) continue;

				resultIndexes.push_back(hostUserBasicIndexes[i] + x);
				selectedRowsCount += 1;
			}
		}

		userStartIndexes[i] = lastIndex;
		lastIndex += selectedRowsCount;
	}

	//for (int i = 0; i < resultIndexes.size(); i++) cout << resultIndexes[i] << " "; cout << endl;
	usersRowsIndexes = (int*)malloc(resultIndexes.size() * sizeof(int));
	copy(resultIndexes.begin(), resultIndexes.end(), usersRowsIndexes);
	return resultIndexes.size();
}