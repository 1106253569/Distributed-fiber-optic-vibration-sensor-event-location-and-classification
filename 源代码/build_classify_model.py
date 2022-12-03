from MyFunction import get_XY, normalization, train_and_test_data
from keras.utils.np_utils import to_categorical
from MyModel import classify_model


if __name__ == "__main__":
    # 获取数据集
    X, Y, event_label = get_XY(step_number=8)
    m = classify_model()
    m.summary()

    for i in range(10):
        print("i={}".format(i))
        X_train, X_test, Y_train, Y_test = train_and_test_data(X, Y, event_label)
        X_train = normalization(X_train)
        X_test = normalization(X_test)
        Y_train = to_categorical(Y_train, len(event_label))
        Y_test = to_categorical(Y_test, len(event_label))
        # 训练分类器
        m = classify_model()
        m.compile(
            loss="categorical_crossentropy", 
            optimizer="adam", metrics=["accuracy"]
        )
        m.fit(
            X_train,
            Y_train,
            epochs=60,
            batch_size=64,
            use_multiprocessing=True,
        )

        # 保存权重
        name = "classify" + str(i)
        m.save_weights("MyModel/result/" + name + ".h5")

    # 保存卷积神经网络模型
    model_json = m.to_json()
    with open("MyModel/classify.json", "w") as json_file:
        json_file.write(model_json)
