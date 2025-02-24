import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load COVID-19 dataset from Our World in Data
df = pd.read_csv('owid-covid-data.csv')

# Select relevant columns
df = df[['location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_vaccinations']]

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Drop rows with missing values
df.dropna(how="any", inplace=True)

# Set style for Seaborn
sns.set_style("whitegrid")

# 1. Time Series Visualization: COVID-19 Cases Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df[df['location'] == 'India'], x='date', y='total_cases', linewidth=2, color='blue')
plt.title('COVID-19 Total Cases in India Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.show()

# 2. Time Series Visualization: New COVID-19 Cases Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df[df['location'] == 'India'], x='date', y='new_cases', linewidth=2, color='red')
plt.title('COVID-19 New Cases in India Over Time')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.xticks(rotation=45)
plt.show()

# 3. Heatmap: Correlation Between Variables
plt.figure(figsize=(10, 6))
sns.heatmap(df[['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_vaccinations']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# 4. Bar Plot: Top 10 Countries by Total Cases
top_countries = df.groupby('location')['total_cases'].max().nlargest(10).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=top_countries, x='total_cases', y='location', palette='Reds_r')
plt.title('Top 10 Countries by Total Cases')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.show()

# 5. Violin Plot: Distribution of Cases Across Countries (Subset to Avoid Overcrowding)
sampled_countries = df[df['location'].isin(top_countries['location'])]  # Limit to top 10 for clarity
plt.figure(figsize=(12, 6))
sns.violinplot(data=sampled_countries, x='location', y='total_cases', palette='Blues')
plt.title('Distribution of COVID-19 Cases Across Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.show()

# 6. Pair Plot: Relationship Between Features
sns.pairplot(df[['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_vaccinations']])
plt.show()

print("Capstone Project on COVID-19 Data Visualization Completed Successfully!")
