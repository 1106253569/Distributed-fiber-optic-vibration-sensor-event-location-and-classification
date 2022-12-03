from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from MyFunction import normalization, get_YN
from MyModel import location_model


if __name__ == "__main__":
    # 获取数据集
    X,Y = get_YN()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=16
    )
    X_train = normalization(X_train)
    X_test = normalization(X_test)
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    # 训练分类器
    m = location_model()
    m.compile(loss="categorical_crossentropy", 
              optimizer="adam", metrics=["accuracy"])
    m.summary()
    m.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=5,
        batch_size=64,
        use_multiprocessing=True,
    )

    model_json = m.to_json()
    # 权重不在json中,只保存网络结构
    with open("MyModel/location.json", "w") as json_file:
        json_file.write(model_json)
    # 保存权重
    m.save_weights("MyModel/location.h5")
