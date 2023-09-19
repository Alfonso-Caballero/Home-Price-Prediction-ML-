from src.utils.utils import pd, train_test_split, Pipeline, GridSearchCV, ColumnTransformer, StandardScaler, OneHotEncoder, ElasticNet, RandomForestRegressor, sns, xgb, MLPRegressor, keras, np, plt, r2_score, mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


def elastic_net(onehot_columns, numerical_columns, x_tr, y_tr, x_te, y_te):

    ct = ColumnTransformer([
        ('standard', StandardScaler(), numerical_columns),
        ('onehot', OneHotEncoder(handle_unknown="ignore"), onehot_columns)
                            ], remainder='passthrough'
                          )

    pipeline = Pipeline([
        ('column_transformer', ct),
        ('elastic', ElasticNet())
                        ])

    param_grid = {'elastic__alpha': [0.00001, 0.0001, 0.001, 0.01],
                  'elastic__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                  }

    grid_pipeline = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=2, refit=True)
    grid_pipeline.fit(x_tr, y_tr)
    y_predict = grid_pipeline.predict(x_te)

    print(np.sqrt(mean_squared_error(y_te, y_predict)))
    print(grid_pipeline.score(x_te, y_te))
    print(grid_pipeline.best_params_)
    print(grid_pipeline.best_score_)


def random_forest(onehot_columns, x_tr, y_tr, x_te, y_te):

    ct = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown="ignore"), onehot_columns)
                            ], remainder='passthrough'
                          )

    pipeline = Pipeline([
        ('column_transformer', ct),
        ('forest', RandomForestRegressor())
                        ])

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    grid_pipeline = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, refit=True)
    grid_pipeline.fit(x_tr, y_tr)
    y_predict = grid_pipeline.predict(x_te)

    df = pd.DataFrame({'Valores Reales': y_te, 'Predicciones': y_predict}).astype(int)
    sns.scatterplot(x=y_te, y=y_predict)
    plt.xlabel('y_test data')
    plt.ylabel('Predictions')
    plt.show()
    print(df)
    print(r2_score(y_te, y_predict))
    print(np.sqrt(mean_squared_error(y_te, y_predict)))


def xgboost(onehot_columns, numerical_columns, x_tr, y_tr, x_te, y_te):

    ct = ColumnTransformer([
        ('standard', StandardScaler(), numerical_columns),
        ('onehot', OneHotEncoder(handle_unknown="ignore"), onehot_columns)
                            ], remainder='passthrough'
                           )

    pipeline = Pipeline([
        ('column_transformer', ct),
        ('boost', xgb.XGBRegressor())
                        ])

    param_grid = {
        'boost__max_depth': [2, 3, 5, 7, 10],
        'boost__n_estimators': [10, 100, 500],
    }

    grid_pipeline = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, refit=True)
    grid_pipeline.fit(x_tr, y_tr)
    y_predict = grid_pipeline.predict(x_te)

    df = pd.DataFrame({'Valores Reales': y_te, 'Predicciones': y_predict}).astype(int)
    sns.scatterplot(x=y_te, y=y_predict)
    plt.xlabel('y_test data')
    plt.ylabel('Predictions')
    plt.show()
    print(df)
    print(r2_score(y_te, y_predict))
    print(grid_pipeline.best_params_)
    print(np.sqrt(mean_squared_error(y_te, y_predict)))
    

def mlp_regressor(onehot_columns, numerical_columns, x_tr, y_tr, x_te, y_te):

    ct = ColumnTransformer([
        ('standard', StandardScaler(), numerical_columns),
        ('onehot', OneHotEncoder(handle_unknown="ignore"), onehot_columns)
                            ], remainder='passthrough'
                          )

    pipeline = Pipeline([
        ('column_transformer', ct),
        ('mlpregressor', MLPRegressor(random_state=42, max_iter=5000, solver='adam', hidden_layer_sizes=(64,), early_stopping=True, n_iter_no_change=10, activation="relu"))
                        ])

    pipeline.fit(x_tr, y_tr)
    y_predict = pipeline.predict(x_te)
    print(np.sqrt(mean_squared_error(y_te, y_predict)))
    print(pipeline.score(x_te, y_te))
    print(pipeline.score(x_tr, y_tr))
    print(print(y_predict[:6]))
    print(y_te.head())


def keras_neural(onehot_columns, numerical_columns, X_train_full, x_test, y_train_full, y_te):

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

    ct = ColumnTransformer([
        ('scaler', StandardScaler(), numerical_columns),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_columns)
                            ], remainder='passthrough'
                           )

    X_train = ct.fit_transform(X_train).toarray()
    X_valid = ct.transform(X_valid).toarray()
    x_test = ct.transform(x_test).toarray()

    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1:])),
        keras.layers.Dense(1)
                                    ])

    optimizer = keras.optimizers.SGD(clipnorm=1)

    model.compile(loss="mean_squared_error", optimizer=optimizer)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.fit(X_train, y_train, epochs=1000, validation_data=(X_valid, y_valid), callbacks=[early_stopping_cb])

    mse_test = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    print(r2_score(y_te, y_predict))
    print(np.sqrt(mse_test))


if __name__ == '__main__':

    train_dataframe = pd.read_csv("data/processed/train_set.csv")
    test_dataframe = pd.read_csv("data/processed/test_set.csv")
    X_train_pre = train_dataframe[["Place", "Location", "Area", "Rooms", "Air Conditioning", "Built-in Wardrobes", "Elevator", "Heating", "Garage", "Terrace"]]
    y_train_pre = train_dataframe["Price"]
    X_test = test_dataframe[["Place", "Location", "Area", "Rooms", "Air Conditioning", "Built-in Wardrobes", "Elevator", "Heating", "Garage", "Terrace"]]
    y_test = test_dataframe["Price"]

    oh_columns = ["Place", "Location"]
    n_columns = ["Area", "Rooms"]

    elastic_net(oh_columns, n_columns, X_train_pre, y_train_pre, X_test, y_test)
    random_forest(oh_columns, X_train_pre, y_train_pre, X_test, y_test)
    xgboost(oh_columns, n_columns, X_train_pre, y_train_pre, X_test, y_test)
    mlp_regressor(oh_columns, n_columns, X_train_pre, y_train_pre, X_test, y_test)
    keras_neural(oh_columns, n_columns, X_train_pre, X_test, y_train_pre, y_test)
