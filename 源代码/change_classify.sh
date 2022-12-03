#! /bin/bash
# 测试分类模型并设置classify.h5

python test_classify_model.py

echo "是否更换模型参数文件(.h5)"
input='N';
if read -t 20 -p "[20秒内未输入默认为N] (Y/N): " input
then
    echo "Countine...";
else
    echo -e "\n抱歉,已超时";
    input='N';
fi

if [ "${input}" == "Y" ] || [ "${input}" == "y" ]; then
    echo "输入新的模型参数文件编号(0-9)"
    i=10
    if read -t 60 -p "[60秒内未输入将不做更改] (i<10): " i
    then
        echo "Countine...";
    else
        echo -e "\n抱歉,已超时";
        i=10
    fi
else
    echo "!= Y/y, 退出更改"
    exit 
fi

if [ $i -gt -1 ] && [ $i -lt 10 ]; then 
    echo "正在更改"
    path="MyModel/result/classify${i}.h5"
    cp -i $path MyModel/classify.h5
    echo "更改完成"
else
    echo "i=$i 不符合条件,退出更改"
    exit 
fi