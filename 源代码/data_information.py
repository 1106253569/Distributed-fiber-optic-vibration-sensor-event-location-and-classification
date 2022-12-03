import numpy as np


class data_information:
    def __init__(self, information) -> None:
        file_name = information["file_name"]
        self.data_path = (
            "/home/T02053124/数据/original_dataset/"
            + file_name.split("\\")[0]
            + "/"
            + file_name.split("\\")[1]
        )
        self.event_local = information["boxes"][0]
        self.data_label = information["labels"]
        self.data_name = information["file_name"].split("\\")[1].split(".")[0]
        self.dataset = np.zeros(0)

    def read_dataset(self):
        path_name = (
            "/home/T02053124/数据/train_list/"
            + self.data_name
            + "_"
            + str(self.data_label[0])
            + ".csv"
        )
        temp = np.loadtxt(path_name, delimiter=",")
        return temp
