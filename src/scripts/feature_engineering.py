from src.utils.utils import pd, train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':

    dataframe = pd.read_csv("../data/raw/houses_info.csv")
    df = dataframe.copy()
    X = df[["Place", "Location", "Area", "Rooms", "Toilets", "Air Conditioning", "Built-in Wardrobes", "Elevator", "Heating", "Garage", "Terrace"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    non_fe_set = pd.concat([X_train, y_train], axis=1)
    non_fe_set.drop(columns=["Toilets"], inplace=True)
    non_fe_set.drop_duplicates(inplace=True, ignore_index=True, keep="first")
    non_fe_set.dropna(inplace=True)
    non_fe_set = non_fe_set[non_fe_set.Area <= 500]
    non_fe_set.reset_index(inplace=True, drop=True)
    non_fe_set.to_csv("../data/processed/train_set.csv", index=False)

    test_set = pd.concat([X_test, y_test], axis=1)
    test_set.reset_index(inplace=True, drop=True)
    test_set.to_csv("../data/processed/test_set.csv", index=False)





