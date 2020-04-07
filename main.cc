#include<stdio.h>
#include<stdlib.h>
#include<math.h>
float operations(int op, float a, float b, float c, int d, char* str);
//double activation(double x) {
//    return 1 / (1 + exp(-x));
//}
// Байт-код для обучения сети
typedef enum {
	RELU,
	RELU_DERIV,
	SIGMOID,
	SIGMOID_DERIV,
	TRESHOLD_FUNC,
	TRESHOLD_FUNC_DERIV,
	LEAKY_RELU,
	LEAKY_RELU_DERIV,
	INIT_W_HE,
	INIT_W_GLOROT,
	INIT_W_STEPAN_M_TEST,
	DEBUG,
	DEBUG_STR
} OPS;
// Логическое ИЛИИ
double data[4][2] = {
	{0, 0},
	{1, 0},
	{0, 1},
	{1, 1}
};
double answer[4] = {0, 1, 1, 1};
double n1[2] = {0, 0};
double n2 = 0;
double w2[3] = {0, 0, 0};
double n2_dot = 0; // взвешенное состояние нейрона(в больших сетях нейронов)
int count = 0;
double A = 7, E, E1, E2, E3;

int main() {
	int choose = 3;
	int eps = 10;
	double mse = 0;
	// b - верхняя граница, a - нижняя граница	
	w2[1] = operations(INIT_W_STEPAN_M_TEST, 3, -0.0025, 0.0025, 0, ""); // случайные веса от 
	w2[2] = operations(INIT_W_STEPAN_M_TEST, 3, -0.0025, 0.0025, 0, "");
	w2[0] = operations(INIT_W_STEPAN_M_TEST, 3, -0.0025, 0.0025, 0, ""); // биас
	//	double q = 0.0; // правильные ответы
	/*while (count < eps)*/ while (1) {
		printf("epocha %d\n", count);
		//		choose = rand() % 3; // случайно выбираю входные данные
		// Рандомно выбирать неправильно - надо ведь все их обойти
		while (choose >= 0) {
			printf("chose %d\n", choose);
			n1[0] = data[choose][0]; // поставляю входные данные в
			n1[1] = data[choose][1]; // нейронах 1 слоя
			/*Умножаю значения нейронов 1 слоя с соответствующими весами и
			  пропускаю через функцию активации которая является сигмоидом*/
			n2_dot = n1[0] * w2[0] + n1[1] * w2[2] + w2[0];
			n2 = operations(SIGMOID, n2_dot, 0.5, 0, 0, "");
			// Получаю ошибку выходного нейрона
			E = (answer[choose] - n2) * operations(SIGMOID_DERIV, n2_dot, 0.5, 0, 0, "");
			mse = pow(answer[choose] - n2, 2);
			printf("mse in train: %f\n", mse);
			if (mse < 0.000100000)
				goto out_bach;
			E1 = E * w2[0];
			E2 = E * w2[1];
			E3 = E * w2[2];
			// изменяю веса по формулу w = w + A* Еrror * fx *(1-fx) * INPUT 
			w2[1] = w2[0] + A * E1 * n1[0];
			w2[2] = w2[1] + A * E2 * n1[1];
			w2[0] = w2[1] + A * E3 * (+1);
			choose--;
		}
		choose = 3;
		count++;
	}
out_bach:
	;
	// Сеть обучилась - проведем консольную кросс-валидацию
	printf("***Cons Cv - Logic Or***\n");
	choose = 3;
	while (choose >= 0) {
		printf("chose %d\n", choose);
		n1[0] = data[choose][0]; // поставляю входные данные в
		n1[1] = data[choose][1]; // нейронах 1 слоя
		/*Умножаю значения нейронов 1 слоя с соответствующими весами и
		  пропускаю через функцию активации которая является сигмоидом*/
		n2_dot = n1[0] * w2[0] + n1[1] * w2[2] + w2[0];
		n2 = operations(SIGMOID, n2_dot, 0.5, 0, 0, "");
		printf("input vector [ %f %f ] ",n1[0], n1[1]);
		if(n2 > 0.5)
			printf("output vector[ %f ]\n",1);
		else
			printf("output vector[ %f ]\n",0);
		choose--;
	}
	system("pause");
	return 0;
}

//-----------------[Операция наподобии виртуальной машины]------------
float operations(int op, float a, float b, float c, int d, char* str) {
	switch (op) {
	case RELU:
	{
		if (a <= 0)
			return 0;
		else
			return a;
	}
	case RELU_DERIV:
	{
		if (a <= 0)
			return 0;
		else
			return 1;
	}
	case TRESHOLD_FUNC:
	{
		if (a < 0)
			return 0;
		else
			return 1;
	}
	case TRESHOLD_FUNC_DERIV:
	{
		return 0;
	}
	case LEAKY_RELU:
	{
		if (a < 0)
			return b * a;
		else
			return a;
	}
	case LEAKY_RELU_DERIV:
	{
		if (a < 0)
			return b;
		else
			return 1;
	}
	case SIGMOID:
	{
		return 1.0 / (1 + exp(b * (-a)));
	}
	case SIGMOID_DERIV:
	{
		return(b * 1.0 / (1 + exp(b * (-a))))*(1 - 1.0 / (1 + exp(b * (-a))));
	}
	case DEBUG:
	{
		printf("%s : %f\n", str, a);
		break;
	}
		// Лучше использовать Гауссовое распределение,я его из Python получил
		//	case INIT_W_HE:
		//	{
		//		PyObject * pVal;
		//		float r = 0;
		//		pVal = PyObject_CallMethod(pInstanceRandom, "gauss", "ii", 0, 1);
		//		if (pVal != NULL) r = PyFloat_AsDouble(pVal), clear_pyObj(pVal), printf("r he:%f\n", r);
		//		else PyErr_Print();
		//		decr(pVal);
		//		return r * sqrt(2 / a);
		//	}
	case INIT_W_STEPAN_M_TEST:
	{
		srand(42);
		// а - cумма нейронов(входные + выходные),b и c - диапазон	
		return float((rand() / RAND_MAX) * (b / a - c / a)) + b / a;
	}
	case DEBUG_STR:
	{
		printf("%s\n", str);
	}
	}
}
//-----------------[Операция наподобии виртуальной машины]------------