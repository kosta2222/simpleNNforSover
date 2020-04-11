from operations_func import operations
from nn_constants import  INIT_W_HE, RELU,RELU_DERIV, INIT_W_HABR, INIT_W_MY, SIGMOID, SIGMOID_DERIV
def main():
    data=[[0,0],[1,0],[0,1],[1,1]]
    # answer=[0, 1, 1, 1]  #  OR
    answer=[0, 0, 0, 1]  # AND
    n1=[0]*2
    n2=0
    w2=[0]*3
    n2_dot=0
    count=0
    A=0.01
    E=0
    E1=0
    E2=0
    E3=0
    choose=0
    choose_cv=0
    eps=10
    mse=0
    exit_flag=False
    scores=[]
    theme=""
    theme="AND"
    # theme="OR"
    alpha=0.99
    beta=1.01
    gama=1.01
    delta_E_spec=0
    Z=0
    Z_t_minus_1=0
    A_t_minus_1=0
    acc=0
    sigmoid_koef=0.42
    accuracy_shureness = 100
    with_adap_lr = False

    w2[0] = operations(INIT_W_HE, 2, 0, 0, 0, ""); # биас
    w2[1] = operations(INIT_W_HE, 2, 0, 0, 0, "");
    w2[2] = operations(INIT_W_MY, 2, 0, 0, 0, "");

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
            n2 = operations(SIGMOID, n2_dot, sigmoid_koef, 0, 0, "")
            # Получаю ошибку выходного нейрона
            Z = n2 - answer[choose]
            E = (answer[choose] - n2) * operations(SIGMOID_DERIV, n2_dot, sigmoid_koef, 0, 0, "");
            if count == 0:
                Z_t_minus_1 = Z
                A_t_minus_1 = A
            mse = pow(answer[choose] - n2, 2);
            print("mse in train: %f \n" % mse);
            # if (mse < 0.0001):
            #     print("op")
            #     exit_flag=True
            #     break
            if with_adap_lr:
                delta_E_spec = Z - gama * Z_t_minus_1
                if delta_E_spec > 0:
                    A = alpha * A_t_minus_1
                else:
                    A = beta * A_t_minus_1
                print("A",A)
            A_t_minus_1 = A
            Z_t_minus_1 = Z

            E1 = E * w2[0];
            E2 = E * w2[1];
            E3 = E * w2[2];
            # изменяю веса
            w2[0] = w2[0] + A * E1 * (+1);
            w2[1] = w2[1] + A * E2 * n1[0];
            w2[2] = w2[2] + A * E3 * n1[1];

            choose_cv = 0;
            while (choose_cv <= 3):
                print("chose %d\n" % choose);
                n1[0] = data[choose][0];  # поставляю входные данные в
                n1[1] = data[choose][1];  # нейронах 1 слоя
                """
                / * Умножаю значения нейронов 1 слоя с соответствующими весами и
                пропускаю через функцию активации которая является сигмоидом * /
                """
                n2_dot = w2[0] + n1[0] * w2[1] + n1[1] * w2[2] ;
                n2 = operations(SIGMOID, n2_dot, sigmoid_koef, 0, 0, "");
                print("input vector [ %f %f ] " % (n1[0], n1[1]));
                if (n2 > 0.5):
                    n2 = 1
                    print("output vector[ %f ] " % 1, end=' ')
                else:
                    n2 = 0
                    print("output vector[ %f ] " % 0, end=' ');
                print("expected [ %f ]\n" % answer[choose_cv]);
                if n2 == answer[choose_cv]:
                    scores.append(1)
                else:
                    scores.append(0)

                choose_cv += 1;
            acc = sum(scores) / 4 * 100
            print("Accuracy statement",acc)
            scores.clear()
            if acc == accuracy_shureness:
                exit_flag = True
                break

            choose+=1;

        choose = 0;
        count+=1;
        if exit_flag:
            break
    scores.clear()
    """
    Сеть
    обучилась - проведем
    консольную
    кросс - валидацию
    """
    print("***Cons Cv - %s***\n" % theme);
    choose = 0;
    while (choose <= 3):
        print("chose %d\n" % choose);
        n1[0] = data[choose][0];  # поставляю входные данные в
        n1[1] = data[choose][1];  # нейронах 1 слоя
        """
        / * Умножаю значения нейронов 1 слоя с соответствующими весами и
        пропускаю через функцию активации которая является сигмоидом * /
        """
        n2_dot = w2[0] + n1[0] * w2[1] + n1[1] * w2[2] ;
        n2 = operations(SIGMOID, n2_dot, sigmoid_koef, 0, 0, "");
        print("input vector [ %f %f ] " % (n1[0], n1[1]));
        if (n2 > 0.5):
            n2 = 1
            print("output vector[ %f ] " % 1, end=' ')
        else:
            n2 = 0
            print("output vector[ %f ] " % 0, end=' ');
        print("expected [ %f ]\n" % answer[choose]);
        if n2 == answer[choose]:
            scores.append(1)
        else:
            scores.append(0)

        choose += 1;
    acc = sum(scores) / 4 * 100
    print("Accuracy", acc)

main()