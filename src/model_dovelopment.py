from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def model_dovelopment(final_data):
    X = final_data.drop('energy_demand', axis=1)
    y = final_data['energy_demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.columns)

    try:
        Lr_Model = LinearRegression()
        Lr_Model.fit(X_train, y_train)
        lr_pred = Lr_Model.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        print("Model Performance:")
        print("Mean Squared Error:", lr_mse)
        print("R2 Score:", lr_r2)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        print("Model Performance:")
        print("Mean Squared Error:", rf_mse)
        print("R2 Score:", rf_r2)

    except Exception as e:
        print(f"Error Occured in {e}")