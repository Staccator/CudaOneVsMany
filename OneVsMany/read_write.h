#include <string.h>
#include <map>
using namespace std;

void ReadData(int &rowCount, int &columnCount, int &userCount, float* &dataTab, int* &userTab,
	int* &userRows, vector<string> &ids, int* &sessionNumbers);

void WriteResults(float* results, int checkedRowsCount, int userCount,
	int* usersRowsIndexes, int* userStartIndexes, vector<string> ids, int* sessionNumbers);

map<string, string> ReadConfig();
void PrintTab(int * tab, int n);
void PrintTab2D(int n, float* tab);
void PrintUsersSelfDistances(int userCount, int* userRows, int* startIndexes, float* distancesUser);
