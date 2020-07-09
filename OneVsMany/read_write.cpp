#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <map>
#include <iterator>
using namespace std;

string InputFileName;
string OutputFileName;
string Delimiter;

void ReadData(int &rowCount, int &columnCount, int &userCount, float* &dataTab, int* &userTab, int* &userRows, vector<string> &ids, int* &sessionNumbers)
{
	const int SkippedColumns = -4;
	columnCount = SkippedColumns;
	rowCount = 0;
	userCount = 0;

	ifstream inputSetup(InputFileName);
	string line;
	getline(inputSetup, line);
	istringstream firstLine(line);
	string label;

	while (getline(firstLine, label, ','))
		columnCount++;

	string last = "";
	while (getline(inputSetup, line)) {
		istringstream is(line);
		string part;
		getline(is, part, ',');
		if (part != last)
			userCount++;
		last = part;
		rowCount++;
	}

	cout << "Columns:" << columnCount << " | Rows:" << rowCount << " | Users:" << userCount << endl;

	int tabMemory = sizeof(float) * rowCount * columnCount;
	dataTab = (float*)malloc(tabMemory);
	int userMemory = sizeof(int) * rowCount;
	userTab = (int*)malloc(userMemory);
	int userRowsMemory = sizeof(int) * userCount;
	userRows = (int*)malloc(userRowsMemory);
	sessionNumbers = (int*)malloc(userMemory);

	ifstream input(InputFileName);
	last = "";
	int currentUser = -1;
	for (int i = -1; getline(input, line); i++)
	{
		if (i == -1) continue;
		istringstream is(line);
		string part;
		for (int j = SkippedColumns; getline(is, part, ','); j++)
		{
			if (j == -3 || j == -1) continue;
			else if (j == SkippedColumns)
			{
				if (last != part)
				{
					ids.push_back(part);
					currentUser++;
					userRows[currentUser] = 0;
				}
				userTab[i] = currentUser;
				userRows[currentUser]++;
				last = part;
			}
			else if (j == -2)
			{
				sessionNumbers[i] = stoi(part);
			}
			else
			{
				float value;
				if (part == "NA") value = 0;
				else if (part == "FALSE") value = 0;
				else if (part == "TRUE") value = 1;
				else value = stod(part);

				dataTab[i * columnCount + j] = value;
			}
		}
	}
}

void WriteResults(float* results, int checkedRowsCount, int userCount,
	int* usersRowsIndexes, int* userStartIndexes, vector<string> ids, int* sessionNumbers)
{
	ofstream outfile;
	outfile.open(OutputFileName);
	for (int i = 0; i < userCount; i++)
	{
		string id = ids[i];
		int start = userStartIndexes[i];
		int end = i == userCount - 1 ? checkedRowsCount : userStartIndexes[i + 1];
		for (int j = start; j < end; j++)
		{
			int mySession = sessionNumbers[usersRowsIndexes[j]];
			outfile << id << Delimiter << mySession;

			for (int k = 0; k < userCount; k++)
			{
				if (k == i) continue;
				string otherId = ids[k];
				int otherStart = userStartIndexes[k];
				int otherEnd = k == userCount - 1 ? checkedRowsCount : userStartIndexes[k + 1];
				for (int l = otherStart; l < otherEnd; l++)
				{
					int otherSession = sessionNumbers[usersRowsIndexes[l]];
					outfile << "," << otherId << Delimiter << otherSession << Delimiter << results[j*checkedRowsCount + l];
				}
			}
			outfile << endl;
		}
	}
	outfile.close();
}

map<string, string> Config;

map<string, string> ReadConfig()
{
	if (!Config.empty()) return Config;

	string ConfigFileName = "../Config.txt";
	ifstream configStream(ConfigFileName);
	string line;
	istringstream firstLine(line);
	string label;

	while (getline(configStream, line)) {
		istringstream is(line);
		string variableName;
		string variableValue;
		getline(is, variableName, '=');
		getline(is, variableValue);
		Config.insert(make_pair(variableName, variableValue));
	}

	//for (auto it = Config.begin(); it != Config.end(); it++)
		//std::cout << it->first << " :: " << it->second << std::endl;

	InputFileName = Config["InputFileName"];
	OutputFileName = Config["OutputFileName"];
	Delimiter = Config["Delimiter"];

	return Config;
}

void PrintTab(int * tab, int n)
{
	for (int i = 0; i < n; i++)
	{
		cout << setw(6) << tab[i] << " ";
	}
	cout << endl;
}

void PrintTab2D(int n, float* tab)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << setw(6) << tab[i * n + j] << " ";
		}
		cout << endl;
	}
}

void PrintUsersSelfDistances(int userCount, int* userRows, int* startIndexes, float* distancesUser)
{
	for (int i = 0; i < userCount; i++)
	{
		int rows = userRows[i];
		int startIndex = startIndexes[i];
		cout << "\nUser: " << i << endl;
		for (int x = 0; x < rows; x++)
		{
			for (int y = 0; y < rows; y++)
			{
				cout << setw(6) << distancesUser[startIndex + x * rows + y] << " ";
				//cout << startIndex + x * rows + y << endl;
			}
			cout << endl;
		}

	}
}
