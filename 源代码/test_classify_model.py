import numpy as np
from MyModel import MyModel
from MyFunction import get_XY


if __name__=='__main__':
    X, Y, event_label = get_XY()

    X_extend, Y_extend = [], []
    i = 0
    for event in event_label:
        while Y[i] == event_label[event]:
            i += 1
            if i == np.size(Y):
                break
        X_extend.append(X[i - 1])
        Y_extend.append(Y[i - 1])

    for i in range(10):
        print("i={}".format(i))
        name = "classify" + str(i) + ".h5"
        my_model = MyModel("classify.json", "result/" + name)
        my_model.evaluate(X, Y)

        j = 0
        for event in event_label:
            X_temp = []
            Y_temp = []
            print(event, end=":")
            while Y[j] == event_label[event]:
                X_temp.append(X[j])
                Y_temp.append(Y[j])
                j += 1
                if j == np.size(Y):
                    break
            X_temp = np.array(X_temp)
            X_temp[-6:] = np.array(X_extend)
            Y_temp[-6:] = np.array(Y_extend)
            my_model.evaluate(X_temp, Y_temp)
            p_temp = my_model.predict(X_temp)
            print(p_temp)
