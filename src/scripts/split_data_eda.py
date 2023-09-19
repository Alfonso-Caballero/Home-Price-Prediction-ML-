from src.utils.utils import pd, train_test_split, sns, plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


def corr_map(df_num):
    sns.heatmap(df_num.corr(), center=1)
    plt.show()


if __name__ == '__main__':

    dataframe = pd.read_csv("../data/raw/houses_info.csv")
    X = dataframe[["Place", "Location", "Rooms", "Toilets", "Area", "Air Conditioning", "Built-in Wardrobes", "Elevator", "Heating", "Garage", "Terrace"]]
    y = dataframe["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    eda_dataframe = pd.concat([X_train, y_train], axis=1)
    eda_dataframe.reset_index(inplace=True, drop=True)

    eda_dataframe.to_csv("../data/raw/split_houses.csv", index=False)

    numeric_df = eda_dataframe[["Area", "Rooms", "Toilets"]]
    print(numeric_df["Area"].var())
    print(numeric_df["Rooms"].var())
    print(numeric_df["Toilets"].var())
    print(y.var())
    corr_map(numeric_df)
    sns.regplot(x="Price", y="Area", data=eda_dataframe)
    plt.show()
