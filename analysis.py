import pandas as pd
import os, re
import seaborn as sns
import matplotlib.pyplot as plt

# model
model_name = "Lenet5"
data_name = "MNIST"
mia_data_path = "./cover_data"
coverage_type = "NC"


def load_df(mia_data_path, model_name, data_name, data_type):
    df = pd.DataFrame(data=None, columns=[data_type])

    pattern = model_name + "-" + data_name + "-" + data_type + "-" + coverage_type + ".*txt"
    re_pattern = re.compile(pattern=pattern)

    file_list = os.listdir(mia_data_path)
    for file_name in file_list:
        if re_pattern.search(file_name):
            frame_slice = pd.read_table(os.path.join(mia_data_path, file_name), header=None)
            frame_slice.columns = [data_type]
            df = pd.concat([df, frame_slice])

    return df


def stat_df(data_frame):
    data_columns = data_frame.columns
    df = pd.DataFrame(data=None, columns=data_columns.values)

    # mean
    df.loc["mean"] = list(data_frame.mean())

    # std
    df.loc["std"] = list(data_frame.std())

    #1/4 2/4 3/4
    df.loc["Q1"] = list(data_frame.quantile(0.25))
    df.loc["Q2"] = list(data_frame.quantile(0.50))
    df.loc["Q3"] = list(data_frame.quantile(0.75))

    print(df)

def fig_df(data_frame):
    sns.displot(data_frame)
    plt.show()


if __name__ == "__main__":
    train_frame = load_df(mia_data_path, model_name, data_name, "train")
    test_frame = load_df(mia_data_path, model_name, data_name, "test")

    data_frame = pd.concat([train_frame, test_frame], axis=1)

    #
    stat_df(data_frame)
    # 此函数需要在Spyder里面才能正常调用
    fig_df(data_frame)
