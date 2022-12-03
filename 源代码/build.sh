#! /bin/bash
# 训练一维卷积神经网络

echo "训练用于定位的一维卷积神经网络"
python build_location_model.py
echo "训练用于分类的一维卷积神经网络"
python build_classify_model.py

echo "训练完毕,测试分类模型"
./change_classify.sh