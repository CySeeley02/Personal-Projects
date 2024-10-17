import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load your dataset from the Excel file
# Replace 'path/to/your_dataset.xlsx' with the full file path of your Excel file
df = pd.read_csv(r'C:\Users\cyase\OneDrive\Documents\Personal Projects\2024 Pitching Stats.csv')

# Question 1: Difference in sweet spot percentage based on walk rates
threshold = df['bb_percent'].median()
high_walk_rate = df[df['bb_percent'] > threshold]
low_walk_rate = df[df['bb_percent'] <= threshold]

# Perform a t-test
t_stat, p_value = ttest_ind(high_walk_rate['sweet_spot_percent'], low_walk_rate['sweet_spot_percent'])

print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a significant difference in sweet spot percentage between high and low walk rate pitchers.")
else:
    print("There is no significant difference in sweet spot percentage between high and low walk rate pitchers.")

# Visualization for Sweet Spot Percentage by Walk Rate
plt.figure(figsize=(12, 6))
sns.boxplot(x='bb_percent', y='sweet_spot_percent', data=df)
plt.title('Sweet Spot Percentage by Walk Rate', fontsize=18)
plt.xlabel('Walk Rate (bb_percent)', fontsize=14)
plt.ylabel('Sweet Spot Percentage', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Question 2: Predicting xwOBA based on various features
features = ['whiff_percent', 'swing_percent', 'sweet_spot_percent', 'hard_hit_percent']
X = df[features]
y = df['xwoba']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Display the coefficients
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print(coefficients)

# Visualization for Predicted vs Actual xwOBA
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.title('Predicted vs Actual xwOBA', fontsize=18)
plt.xlabel('Actual xwOBA', fontsize=14)
plt.ylabel('Predicted xwOBA', fontsize=14)
plt.tight_layout()
plt.show()

# Visualization for the relationship between strikeout rate and other metrics
plt.figure(figsize=(14, 10))  # Increased figure size for better visibility
plt.subplot(3, 1, 1)
sns.scatterplot(x='swing_percent', y='k_percent', data=df)
plt.title('Strikeout Rate vs Swing Percentage', fontsize=18)
plt.xlabel('Swing Percentage', fontsize=14)
plt.ylabel('Strikeout Rate (k_percent)', fontsize=14)

plt.subplot(3, 1, 2)
sns.scatterplot(x='whiff_percent', y='k_percent', data=df)
plt.title('Strikeout Rate vs Whiff Percentage', fontsize=18)
plt.xlabel('Whiff Percentage', fontsize=14)
plt.ylabel('Strikeout Rate (k_percent)', fontsize=14)

plt.subplot(3, 1, 3)
sns.scatterplot(x='hard_hit_percent', y='k_percent', data=df)
plt.title('Strikeout Rate vs Hard Hit Percentage', fontsize=18)
plt.xlabel('Hard Hit Percentage', fontsize=14)
plt.ylabel('Strikeout Rate (k_percent)', fontsize=14)

plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Question 1: Predicting strikeout rate (k_percent)
X1 = df[['swing_percent', 'whiff_percent', 'hard_hit_percent']]
y1 = df['k_percent']
X1 = sm.add_constant(X1)  # Adds a constant term to the predictor

model1 = sm.OLS(y1, X1).fit()
print("Question 1: Strikeout Rate Prediction")
print(model1.summary())

# Question 2: Relationship between average best speed and ERA
df['ERA'] = np.random.rand(len(df)) * 5  # Replace with your actual ERA data
X2 = df[['avg_best_speed']]
y2 = df['ERA']
X2 = sm.add_constant(X2)

model2 = sm.OLS(y2, X2).fit()
print("\nQuestion 2: Average Best Speed and ERA Relationship")
print(model2.summary())

# Question 3: Correlation between sweet spot percentage and batting average against
df['batting_average_against'] = np.random.rand(len(df))  # Replace with your actual data
X3 = df[['sweet_spot_percent']]
y3 = df['batting_average_against']
X3 = sm.add_constant(X3)

model3 = sm.OLS(y3, X3).fit()
print("\nQuestion 3: Sweet Spot Percentage and Batting Average Against Relationship")
print(model3.summary())

# Question 4: Predicting xwOBA
X4 = df[['whiff_percent', 'swing_percent', 'hard_hit_percent']]
y4 = df['xwoba']
X4 = sm.add_constant(X4)

model4 = sm.OLS(y4, X4).fit()
print("\nQuestion 4: xwOBA Prediction")
print(model4.summary())

# Question 5: Analyzing factors influencing barrel-batted rate
X5 = df[['whiff_percent', 'swing_percent', 'sweet_spot_percent']]
y5 = df['barrel_batted_rate']  # Assuming barrel_batted_rate is present in your dataset
X5 = sm.add_constant(X5)

model5 = sm.OLS(y5, X5).fit()
print("\nQuestion 5: Factors Influencing Barrel-Batted Rate")
print(model5.summary())
