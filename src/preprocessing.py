from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(train, test, falhas):
    # Seleciona apenas colunas numéricas
    X = train.drop(columns=falhas + ["id"]).select_dtypes(include='number')
    
    # Corrige y para ser 0/1
    y = train[falhas].copy()
    for col in y.columns:
        y[col] = y[col].map(lambda x: 1 if x in [1, True, "True"] else 0)

    X_test = test.drop(columns=["id"]).select_dtypes(include='number')

    # Imputação de valores faltantes (com média)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)

    # Escalonamento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    feature_names = train.drop(columns=falhas + ["id"]).select_dtypes(include='number').columns.tolist()

    return X_scaled, y, X_test_scaled, feature_names
