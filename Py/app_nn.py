from operations_func import operations
from nn_constants import  INIT_W_HE, RELU,RELU_DERIV
def main():
    data=[[0,0],[1,0],[0,1],[1,1]]
    answer=[0,1,1,1]
    n1=[0]*2
    n2=0
    w2=[0]*3
    n2_dot=0
    count=0
    A=0.07
    E=0
    E1=0
    E2=0
    E3=0
    choose=0
    eps=10
    mse=0
    exit_flag=False


    w2[0] = operations(INIT_W_HE, 2, 0, 0, 0, ""); # биас
    w2[1] = operations(INIT_W_HE, 2, 0, 0, 0, ""); # случайные
    w2[2] = operations(INIT_W_HE, 2, 0, 0, 0, "");

    while (1) :
        print("epocha %d\n" % count);
        """
        choose = rand() % 3;  случайно выбираю входные данные
         Рандомно выбирать неправильно - надо ведь все их обойти
        """
        while (choose <= 3):
            print("chose %d \n" % choose)
            n1[0] = data[choose][0]; # поставляю входные данные в
            n1[1] = data[choose][1]; # нейронах 1 слоя
            """
            Умножаю значения нейронов 1 слоя с соответствующими весами и
            пропускаю через функцию активации которая является сигмоидом 
            """
            n2_dot = w2[0] + n1[0] * w2[1] + n1[1] * w2[2]
            n2 = operations(RELU, n2_dot, 0, 0, 0, "")
            # Получаю ошибку выходного нейрона
            E = (answer[choose] - n2) * operations(RELU_DERIV, n2_dot, 0, 0, 0, "");
            mse = pow(answer[choose] - n2, 2);
            print("mse in train: %f \n" % mse);
            if (mse < 0.0001):
                print("op")
                exit_flag=True
                break
            E1 = E * w2[0];
            E2 = E * w2[1];
            E3 = E * w2[2];
            # изменяю веса
            w2[0] = w2[0] + A * E1 * (+1);
            w2[1] = w2[1] + A * E2 * n1[0];
            w2[2] = w2[2] + A * E3 * n1[1];

            choose+=1;

        choose = 0;
        count+=1;
        if exit_flag:
            break



    """
    Сеть
    обучилась - проведем
    консольную
    кросс - валидацию
    """
    print("***Cons Cv - Logic Or***\n");
    choose = 0;
    while (choose <= 3):
        print("chose %d\n" % choose);
        n1[0] = data[choose][0]; # поставляю входные данные в
        n1[1] = data[choose][1]; # нейронах 1 слоя
        """
        / * Умножаю значения нейронов 1 слоя с соответствующими весами и
        пропускаю через функцию активации которая является сигмоидом * /
        """
        n2_dot = n1[0] * w2[0] + n1[1] * w2[2] + w2[0];
        n2 = operations(RELU, n2_dot, 0.5, 0, 0, "");
        print("input vector [ %f %f ] "%( n1[0], n1[1]));
        if (n2 > 0.5):
           print("output vector[ %f ] " % 1);
        else:
           print("output vector[ %f ] " % 0);
        print("expected [ %f ]\n"% answer[choose]);
        choose+=1;

main()