import numpy as np
from MyFunction import normalization
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json, Model
from keras.backend import clear_session
from keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D


class MyModel:
    def __init__(self, model_name, h5_name) -> None:
        with open("MyModel/" + model_name, "r") as model:
            json_str = model.read()
        self.model = model_from_json(json_str)
        self.model.load_weights("MyModel/" + h5_name)

    def evaluate(self, X, Y):
        X = normalization(X)
        Y = to_categorical(Y)
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        # 分类准确率
        loss, accuracy = self.model.evaluate(X, Y, verbose=0)
        print("accuracy: {:.2f}%".format(accuracy * 100), end=" - ")
        print("loss: {:.3f}".format(loss))

    def predict(self, X):
        X = normalization(X)
        p = np.argmax(self.model.predict(X, batch_size=32, verbose=0), axis=1)
        return p


def location_model():
    clear_session()
    # 输入层
    inputs = Input(shape=(250, 1))  
    # 卷积层起始
    conv = Conv1D(16, 3, activation="relu")(inputs)
    conv = Conv1D(16, 3, activation="relu")(conv)
    pool = MaxPooling1D(4)(conv)
    conv = Conv1D(64, 3, activation="relu")(pool)
    conv = Conv1D(64, 3, activation="relu")(conv)
    pool = MaxPooling1D(4)(conv) 
    # 卷积层结束
    flatten = Flatten()(pool)
    # 输出层
    output = Dense(2, activation="sigmoid")(flatten)  
    model = Model(inputs=[inputs], outputs=output)
    return model


def classify_model():
    clear_session()
    inputs = Input(shape=(16 * 22, 1))  # 输入层
    # 卷积层起始
    conv = Conv1D(16, 3, activation="relu")(inputs)
    conv = Conv1D(16, 3, activation="relu")(conv)
    pool = MaxPooling1D(3)(conv)
    conv = Conv1D(128, 3, activation="relu")(pool)
    conv = Conv1D(128, 3, activation="relu")(conv)
    pool = MaxPooling1D(3)(conv)
    conv = Conv1D(128, 3, activation="relu")(pool)
    conv = Conv1D(128, 3, activation="relu")(conv)
    pool = MaxPooling1D(3)(conv)
    # 卷积层结束
    flatten = Flatten()(pool)
    output = Dense(6, activation="softmax")(flatten)  # 输出层
    model = Model(inputs=[inputs], outputs=output)
    return model