import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/nandukhanal/Documents/Roadmap Data Science/week-2/2015.csv")
info = df.info()
print(info)

print("\n First 5 rows")
print(df.head())

print("\n Last 5 rows")
print(df.tail())

print("\n Shape ")
print(df.shape)

print("\n Data Columns")
print(df.columns)

print("\n Check the data is null")
print(df.isnull().sum())

mean = df["Economy (GDP per Capita)"].mean()
print("Economy mean is: ", mean)

median = df["Economy (GDP per Capita)"].median()
print("Economy median is",median)

std = df["Economy (GDP per Capita)"].std()
print("Economy std is:", std)

variance = df["Economy (GDP per Capita)"].var()
print("Economy variance is:", variance)


# for health
mean_health = df["Health (Life Expectancy)"].mean()
print("Health mean is: ", mean_health)

median_health = df["Health (Life Expectancy)"].median()
print("Health median is: ", median_health)

std_health = df["Health (Life Expectancy)"].std()
print("Health std is: ", std_health)

var_health = df["Health (Life Expectancy)"].var()
print("Health var is: ", var_health)

# for generosity
mean_genoristy = df["Generosity"].mean()
print("genorosity mean is: ",mean_genoristy)

median_genoristy = df["Generosity"].median()
print("genorosity median is: ",median_genoristy)

std_genorosity = df["Generosity"].std()
print("genorosity std is: ",std_genorosity)

var_genorosity = df["Generosity"].var()
print("genorosity var is: ",var_genorosity)

# Data visualization
# Scatter plot 
plt.figure(figsize=[10,6])

plt.title("Relation between happiness score, GDP, Health and generosity")
plt.scatter(df["Economy (GDP per Capita)"], df["Happiness Score"], label= "Economy", color = "red", alpha=0.7)
plt.scatter(df["Health (Life Expectancy)"], df["Happiness Score"], label= "Health", color = "green", alpha=0.7)
plt.scatter(df["Generosity"], df["Happiness Score"], label= "Generosity", color = "blue", alpha=0.7)

plt.xlabel("Feature value")
plt.ylabel("Happiness score")

plt.legend()
plt.grid(True)

plt.show()

selected_data = df[["Economy (GDP per Capita)","Generosity", "Happiness Score","Health (Life Expectancy)"]]

plt.figure(figsize=[10,6])
sns.heatmap(selected_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=[10,6])
plt.scatter(df["Generosity"], df["Happiness Score"], label= "Generosity", alpha = 0.7, color = "red")
plt.xlabel("Happiness score")
plt.ylabel("Generosity")
plt.title("Outliners for generosity")

plt.legend()
plt.grid(True)

plt.show()

# split data for training 
x = df[["Economy (GDP per Capita)","Generosity","Health (Life Expectancy)"]]
y = df["Happiness Score"]

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluate Performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"The r2_score is: {r2}")
print(f"mean squared error {mse}")

#Final Visualization
plt.figure(figsize=[10,6])
plt.scatter(y_test, y_pred, color = "red", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Happiness score")
plt.ylabel("Predicted Happiness score")
plt.title("Actual vs predicted Happiness score")
plt.show()
