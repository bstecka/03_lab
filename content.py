# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

def sigmoid(x):
    '''
    :param x: wektor wejsciowych wartosci Nx1
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1
    '''
    return 1 / (1 + np.exp(-x))


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    '''
    N = x_train.shape[0]
    sig_n = sigmoid(x_train @ w)
    L_w = np.sum(np.divide(y_train * np.log(sig_n) + (1 - y_train) * np.log(1 - sig_n), -N)) #czemu sumujemy??
    grad = x_train.transpose() @ (sig_n - y_train) / N
    return L_w, grad


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_valus jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    '''
    func_values = []
    w = w0
    _, grad_w = obj_fun(w)
    w = w - eta * grad_w

    for k in range(1, epochs):
        L_w, grad_w = obj_fun(w)
        func_values.append(L_w)
        w = w - eta * grad_w

    L_w, _ = obj_fun(w)
    func_values.append(L_w)
    return w, np.reshape(np.array(func_values), (epochs, 1))


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w,x,y), gdzie x,y oznaczaja podane
    podzbiory zbioru treningowego (mini-batche)
    :param x_train: dane treningowe wejsciowe NxM
    :param y_train: dane treningowe wyjsciowe Nx1
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymal
    ny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu. Wartosci
    funkcji do func_values sa wyliczane dla calego zbioru treningowego!
    '''
    Nb = mini_batch
    M = y_train.shape[0] // Nb #floor division
    func_values = []
    x_batches = []
    y_batches = []

    for m in range(M):
        x_batches.append(x_train[m * Nb: (m + 1) * Nb, :]) #take matrix slice of data (size = Nb * x_train.shape[1])
        y_batches.append(y_train[m * Nb: (m + 1) * Nb, :])

    w = w0
    for k in range(epochs):
        for m in range(M):
            _, grad_w = obj_fun(w, x_batches[m], y_batches[m])
            w = w - eta * grad_w
        L_w, _ = obj_fun(w, x_train, y_train)
        func_values.append(L_w)
    #Funkcja dodatkowo ma zwracac wartosci funkcji celu wywołanej dla całego zbioru treningowego dla kazdej iteracji algorytmu.
    return w, np.reshape(np.array(func_values), (epochs, 1))


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    '''
    w_sliced = w[1:]
    reg = regularization_lambda/2*(np.linalg.norm(w_sliced)**2)
    N = x_train.shape[0]
    sig_n = sigmoid(x_train @ w)
    L_w = np.sum(np.divide(y_train * np.log(sig_n) + (1 - y_train) * np.log(1 - sig_n), -N))
    w_zero = w.copy()
    w_zero[0] = 0
    grad = (x_train.transpose() @ (sig_n - y_train)) / N + regularization_lambda * w_zero
    return L_w + reg, grad


def prediction(x, w, theta):
    '''
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    '''
    sig = sigmoid(x @ w)
    return np.greater(sig, theta).astype(int)


def f_measure(y_true, y_pred):
    '''
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    '''
    TP = np.sum(np.bitwise_and(y_true, y_pred))
    FP = np.sum(np.bitwise_and(np.bitwise_not(y_true), y_pred))
    FN = np.sum(np.bitwise_and(y_true, np.bitwise_not(y_pred)))
    return (2 * TP) / (2 * TP + FP + FN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: ciag treningowy wejsciowy NxM
    :param y_train: ciag treningowy wyjsciowy Nx1
    :param x_val: ciag walidacyjny wejsciowy Nval x M
    :param y_val: ciag walidacyjny wyjsciowy Nval x 1
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini batcha
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :param thetas: lista wartosci progow klasyfikacji theta, ktore maja byc sprawdzone
    :return: funckja wykonuje selekcje modelu. Zwraca krotke (regularization_lambda, theta, w, F), gdzie regularization_lambda
    to najlpszy parametr regularyzacji, theta to najlepszy prog klasyfikacji, a w to najlepszy wektor parametrow modelu.
    Dodatkowo funkcja zwraca macierz F, ktora zawiera wartosci miary F dla wszystkich par (lambda, theta). Do uczenia nalezy
    korzystac z algorytmu SGD oraz kryterium uczenia z regularyzacja l2.
    '''
    best_lambda = 0
    best_theta = 0
    best_w = 0
    best_f = -1
    L = len(lambdas)
    T = len(thetas)
    F = np.zeros(shape=(L, T))
    w = w0
    for l in range(L):
        def funkcja_celu(a, b, c):
            return regularized_logistic_cost_function(a, b, c, lambdas[l])
        w, _ = stochastic_gradient_descent(funkcja_celu, x_train, y_train, w0, epochs, eta, mini_batch)
        for t in range(T):
            f_val = f_measure(y_val, prediction(x_train, w, thetas[t]))
            F[l, t] = f_val
            if f_val >= best_f: #Zwróc wartosci lambda i theta, dla których wartosc F-measure była najwieksza.
                best_f = f_val
                best_lambda = lambdas[l]
                best_theta = thetas[l]
                best_w = w
    return best_lambda, best_theta, best_w, F
