void PrintSelectedRows(int checkedRowsCount, int userCount, int* usersRowsIndexes, int* userStartIndexes);
int* ChooseAll(int rowCount);
int ChooseRandom(int* &usersRowsIndexes, int* &userStartIndexes, int* hostUserRows, int* hostUserBasicIndexes, int userCount);
int ChooseHeuristic(int* &usersRowsIndexes, int* &userStartIndexes, int userCount, int* userRows, int* startIndexes, float* distancesUser, int* hostUserBasicIndexes);
int ChooseWithoutDistant(int* &usersRowsIndexes, int* &userStartIndexes, int userCount, int* userRows, int* startIndexes, float* distancesUser, int* hostUserBasicIndexes);
int ChooseCharacteristicAreas(int* &usersRowsIndexes, int* &userStartIndexes, int userCount, int* userRows, int* startIndexes, float* distancesUser, int* hostUserBasicIndexes);

void LoadChoiceMethodsConfigValues();