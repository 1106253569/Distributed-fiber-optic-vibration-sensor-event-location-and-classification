from location_and_classify_model import location_and_classify_model

if __name__=='__main__':
    i=1
    model = location_and_classify_model()
    while True:
        #实验文件
        #/home/T02053124/数据/original_dataset/大风/220913_142406.xlsx
        #/home/T02053124/数据/original_dataset/电钻/221008_133129.xlsx
        path = input("输入文件名(.xlsx):")
        result = model.location_and_classify(model.read_data(path))
        print("判断结果:",end="\t")
        print(result)
        npy_name="/home/T02053124/数据/result/result"+str(i)+".npy"
        model.save_result(result,path,npy_name)
        c=input("是否继续(Y/N): ")
        if c!='Y' and c!='y':
            break
        i+=1
        
