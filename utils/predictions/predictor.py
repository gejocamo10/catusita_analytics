import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class Predictor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, data_path):
        df = pd.read_csv(data_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values(by=['articulo', 'fecha'])
        return df

    def load_covariates_data(self, cov_path):
        return pd.read_csv(cov_path)

    def create_monthly_sales_data(self, df):
        df.loc[:, 'year'] = df['fecha'].dt.year
        df.loc[:, 'month'] = df['fecha'].dt.month

        monthly_data = df.groupby(['articulo', 'year', 'month']).agg({
            'cantidad': 'sum',
            'transacciones': 'sum',
            'venta_pen': 'sum',
            'fuente_suministro': 'first',
            'lt': 'first'
        }).reset_index()

        monthly_data['fecha'] = pd.to_datetime(monthly_data.apply(
            lambda row: pd.Timestamp(year=int(row['year']), month=int(row['month']), day=1), axis=1))
        monthly_data['lt'] = monthly_data['lt'].fillna(0)
        return monthly_data.sort_values(by=['articulo', 'fecha'])

    def select_features_with_lasso(self, X, y, feature_columns, alpha=0.01):
        """
        Perform feature selection using LASSO regression.
        Returns selected feature names based on non-zero coefficients.
        """
        X_scaled = self.scaler.fit_transform(X)
        
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        selected_features = [feature for feature, coef in zip(feature_columns, lasso.coef_) 
                        if abs(coef) > 0]
                
        return selected_features

    def prepare_features_for_ml(self, all_monthly_data, df_cov, df_correlaciones_sig, sku):
        sku_correlations = df_correlaciones_sig[df_correlaciones_sig['sku'] == sku]
        
        if len(sku_correlations) == 0:
            return all_monthly_data[all_monthly_data['articulo'] == sku].copy(), []
            
        data = all_monthly_data[all_monthly_data['articulo'] == sku].merge(
            df_cov, on=['year', 'month'], how='left'
        )
        
        feature_columns = []
        for _, row in sku_correlations.iterrows():
            col_name = row['tipo']
            lag = row['lag']
            if lag > 0:
                data[f'{col_name}_lag_{lag}'] = data[col_name].shift(lag)
                feature_columns.append(f'{col_name}_lag_{lag}')
            else:
                feature_columns.append(col_name)
        
        data = data.dropna()
        
        if len(data) > 0 and len(feature_columns) > 0:
            X = data[feature_columns]
            y = data['cantidad']
            selected_features = self.select_features_with_lasso(X, y, feature_columns)
            return data, selected_features
        
        return data, []

    def weighted_mape(self, y_true, y_pred):
        errors = []
        for true, pred in zip(y_true, y_pred):
            if pd.isna(true) or pd.isna(pred) or true == 0:
                continue
            error = abs((true - pred) / true)
            if pred < true:  # underprediction
                error *= 2
            errors.append(error)
        return np.mean(errors) * 100 if errors else float('inf')

    def ES_forecast(self, series, alpha):
        series = np.array(series)
        result = [series[0]]
        for n in range(1, len(series)):
            result.append(alpha * series[n] + (1 - alpha) * result[n-1])
        return result[-1]

    def ES_opt_alpha(self, series):
        series = np.array(series)
        alpha_list = np.linspace(0.1, 0.9, 100)
        errors = []
        
        for alpha in alpha_list:
            error = []
            for i in range(1, len(series)-1):
                forecast = self.ES_forecast(series[:i], alpha)
                error.append(abs(forecast - series[i+1]))
            errors.append(np.mean(error))
        
        return alpha_list[np.argmin(errors)]

    def evaluate_models(self, data, feature_columns, target_col='cantidad', 
                    lookback_periods=[6, 12, None], val_year=None):
        best_score = float('inf')
        best_config = None
        best_model = None
        
        if len(data) < 4:
            return None, None, float('inf')
            
        val_data = data[data['year'] == val_year] if val_year else data
        val_size = 6
        
        for lookback in lookback_periods:
            train_data = (val_data.iloc[-(lookback+val_size):-val_size] 
                        if lookback else val_data.iloc[:-val_size])
            test_data = val_data.iloc[-val_size:]
            
            # if len(train_data) < 2:
            #     continue
                
            y_train = train_data['cantidad']
            y_test = test_data['cantidad']
            
            models = {
                'mean': y_train.mean(),
                'median': y_train.median(),
                'es': (self.ES_forecast, self.ES_opt_alpha(y_train))
            }
            
            if len(feature_columns) > 0:
                models.update({
                    'xgboost': xgb.XGBRegressor(random_state=42),
                    'linear': LinearRegression()
                })
                X_train = train_data[feature_columns]
                X_test = test_data[feature_columns]
            
            for model_name, model in models.items():
                try:
                    if model_name in ['xgboost', 'linear']:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_name == 'es':
                        forecast_func, alpha = model
                        y_pred = []
                        current_data = y_train.values
                        for _ in range(len(y_test)):
                            pred = forecast_func(current_data, alpha)
                            y_pred.append(pred)
                            current_data = np.append(current_data, pred)
                    else:  # mean or median
                        y_pred = [model] * len(y_test)
                    
                    score = self.weighted_mape(y_test, y_pred)
                    
                    if score < best_score:
                        best_score = score
                        best_config = (model_name, lookback)
                        best_model = model
                        
                except Exception as e:
                    continue
        
        return best_config, best_model, best_score

    def predict_future_months(self, data, feature_columns, best_model, 
                            best_model_name, lookback, num_months):
        predictions = []
        current_data = data.copy()
        
        for _ in range(num_months):
            if best_model_name in ['xgboost', 'linear']:
                pred = best_model.predict(current_data[feature_columns].iloc[-1:])[0]
            elif best_model_name in ['mean', 'median']:
                pred = best_model
            else:  # ES
                forecast_func, alpha = best_model
                lookback_data = (current_data.iloc[-lookback:] if lookback 
                               else current_data)
                pred = forecast_func(lookback_data['cantidad'].dropna(), alpha)
            
            predictions.append(pred)
            
            new_row = current_data.iloc[-1:].copy()
            new_row['cantidad'] = pred
            new_row['fecha'] += pd.DateOffset(months=1)
            new_row['month'] = new_row['fecha'].dt.month
            new_row['year'] = new_row['fecha'].dt.year
            
            current_data = pd.concat([current_data, new_row])
            
            if feature_columns:
                for lag in range(1, 4):
                    current_data[f'cantidad_lag_{lag}'] = current_data['cantidad'].shift(lag)
            
        return predictions

    def make_final_predictions(self, all_monthly_data, df_cov, df_correlaciones_sig):
        results = []
        no_sku_process_list = []
        count = 0
        for sku in all_monthly_data['articulo'].unique():
            print(f"Processing SKU: {sku}")
            try:
                sku_data = all_monthly_data[all_monthly_data['articulo'] == sku].copy()
                last_date = sku_data['fecha'].max()
                last_date = (last_date + pd.offsets.MonthBegin(1)).normalize()
                lt = int(sku_data['lt'].iloc[-1])
                last_year = sku_data['year'].max()

                # if len(sku_data) < 7:
                #     continue

                data, feature_columns = self.prepare_features_for_ml(
                    all_monthly_data, df_cov, df_correlaciones_sig, sku
                )

                best_config, best_model, score = self.evaluate_models(
                    data,
                    feature_columns,
                    lookback_periods=[6, 12, None],
                    val_year=last_year - 1
                )

                # if best_config is None:
                #     continue

                if best_config is None or best_model is None:
                    results.append({
                        'sku': sku,
                        'lt': lt,
                        'date': last_date,
                        'model': np.nan,
                        'real': 0,
                        'catusita': np.nan,
                        'lookback_period': np.nan,
                        'features_used': 'none',
                        'caa': np.nan,
                        'caa_lt': np.nan,
                        'corr_sd': np.nan,
                        'loss': np.nan
                    })
                    print(f"SKU {sku} no pudo ser evaluado. Datos insuficientes o problema en los datos.")
                    no_sku_process_list.append({'sku':sku})
                    # pd.DataFrame(no_sku_process_list,columns=['sku']).to_csv('data/cleaned/no_sku_process_list.csv')
                    count = count + 1
                    continue

                best_model_name, lookback = best_config

                # Calculate test score
                test_data = data[data['year'] == last_year]
                if feature_columns:
                    test_X = test_data[feature_columns]
                    if best_model_name in ['xgboost', 'linear']:
                        test_pred = best_model.predict(test_X)
                else:
                    if best_model_name in ['mean', 'median']:
                        test_pred = [best_model] * len(test_data)
                    else:  # ES
                        forecast_func, alpha = best_model
                        test_pred = []
                        current_data = data[data['year'] < last_year]['cantidad'].values
                        for _ in range(len(test_data)):
                            pred = forecast_func(current_data, alpha)
                            test_pred.append(pred)
                            current_data = np.append(current_data, pred)

                test_score = self.weighted_mape(test_data['cantidad'], test_pred)

                # Generate future predictions
                future_predictions = self.predict_future_months(
                    data,
                    feature_columns,
                    best_model,
                    best_model_name,
                    lookback,
                    2 * lt
                )

                # Calculate pred_std from the historical data used for prediction
                if lookback:
                    historical_data = data['cantidad'].iloc[-lookback:]
                else:
                    historical_data = data['cantidad']
                pred_std = np.std(historical_data)

                period1 = sum(future_predictions[:lt])
                period2 = sum(future_predictions[lt:2*lt])

                # period2 += 0.4 * pred_std

                last_six_months_mean = sku_data.tail(6)['cantidad'].mean()

                results.append({
                    'sku': sku,
                    'lt': lt,
                    'date': last_date,
                    'model': best_model_name,
                    'real': 0,
                    'catusita': last_six_months_mean,
                    'lookback_period': lookback if lookback else 'all',
                    'features_used': ','.join(feature_columns) if feature_columns else 'none',
                    'caa': period1,
                    'caa_lt': period2,
                    'corr_sd': pred_std,
                    'loss': test_score
                })

            except Exception as e:
                print(f"Error processing SKU {sku}: {str(e)}")
                continue
        print(f"El numero de SKU sin procesar: {count}")
        return pd.DataFrame(results)

    def process_predictions(self):
        from utils.process_data.config import DATA_PATHS
        
        data_path = DATA_PATHS['process'] / 'catusita_consolidated.csv'
        cov_path = DATA_PATHS['process'] / 'df_covariables.csv'
        
        all_monthly_data = self.create_monthly_sales_data(
            self.load_and_preprocess_data(data_path)
        )
        df_cov = self.load_covariates_data(cov_path)
        df_correlaciones_sig = pd.read_csv(DATA_PATHS['process'] / 'df_correlaciones_sig.csv')
        
        results_df = self.make_final_predictions(
            all_monthly_data, 
            df_cov, 
            df_correlaciones_sig
        )

        # results_df = results_df[results_df['date'] == results_df['date'].max()]

        if not results_df.empty:
            return results_df.sort_values('loss')
        return None
