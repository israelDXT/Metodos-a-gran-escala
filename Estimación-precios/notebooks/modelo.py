import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Cargar datos
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
data = pd.read_csv(url)

# Exploración inicial
print(data.head())
print(data.info())
print(data.describe())

# Visualización de relaciones
sns.pairplot(data[['price', 'sqft_living', 'bedrooms', 'bathrooms']])
plt.show()

# Manejo de valores faltantes
imputer = SimpleImputer(strategy='median')

# Codificación de variables categóricas
cat_encoder = OneHotEncoder()

# Escalado de variables numéricas
num_scaler = StandardScaler()

# Definir columnas numéricas y categóricas
num_cols = ['sqft_living', 'bedrooms', 'bathrooms']
cat_cols = ['zipcode']

# Crear pipeline de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', imputer), ('scaler', num_scaler)]), num_cols),
        ('cat', cat_encoder, cat_cols)
    ]
)

# Aplicar preprocesamiento
X = data.drop('price', axis=1)
y = data['price']
X_preprocessed = preprocessor.fit_transform(X)

# Seleccionar las 10 características más relevantes
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_preprocessed, y)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Evaluar
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


def predict_price(sqft_living, bedrooms, bathrooms, zipcode):
    """Predict the price of a house based on its features."""
    # Crear DataFrame con las características proporcionadas
    input_data = pd.DataFrame({
        'sqft_living': [sqft_living],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'zipcode': [zipcode]
    })

    # Preprocesar y predecir
    input_preprocessed = preprocessor.transform(input_data)
    input_selected = selector.transform(input_preprocessed)
    prediction = model.predict(input_selected)
    return prediction[0]


# Ejemplo de uso
predicted_price = predict_price(2000, 3, 2, 98001)
print(f"Precio estimado: ${predicted_price:.2f}")