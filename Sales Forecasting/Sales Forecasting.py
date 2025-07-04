import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Online Sales Data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
print(df.head())
df = df.dropna()
daily_sales = df.groupby('Date')['Total Revenue'].sum().reset_index()
daily_sales['Days'] = (daily_sales['Date'] - daily_sales['Date'].min()).dt.days
plt.figure(figsize=(10, 5))
plt.plot(daily_sales['Date'], daily_sales['Total Revenue'], label='Total Revenue')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.title('Daily Revenue Over Time')
plt.legend()
plt.tight_layout()
plt.show()
X = daily_sales[['Days']]
y = daily_sales['Total Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Days')
plt.ylabel('Revenue')
plt.title('Actual vs Predicted Revenue')
plt.legend()
plt.tight_layout()
plt.show()
days_of_forecast=60
future_days = pd.DataFrame({'Days': list(range(daily_sales['Days'].max() + 1, daily_sales['Days'].max() + days_of_forecast+1))})
future_preds = model.predict(future_days)
last_date = daily_sales['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_of_forecast)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Revenue': future_preds
})
#forecast_df.to_csv('D:\\Data science and Analytics\\forecast_results.csv',index=False)
#print(forecast_df)
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['Date'], daily_sales['Total Revenue'], label='Historical Revenue', marker='o')
plt.plot(forecast_df['Date'], forecast_df['Predicted Revenue'], label='Forecasted Revenue', linestyle='--', marker='x', color='red')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Historical and Forecasted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


