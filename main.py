import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def get_filtered_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
	"""
	Filter outliers from a DataFrame based on a specified column.

	Parameters:
	- df (pd.DataFrame): The DataFrame to filter.
	- column (str): The name of the column to filter on.

	Returns:
	- pd.DataFrame: The filtered DataFrame with outliers removed.

	This function calculates the quartiles and interquartile range (IQR) of the specified column in the DataFrame.
	It then defines the lower and upper bounds for detecting outliers based on the IQR.
	The DataFrame is filtered to remove rows where the values in the specified column fall outside of the defined bounds.
	The filtered DataFrame is returned.
	"""
	# Calcular los cuartiles y el rango intercuartil
	Q1 = df[column].quantile(0.25)
	Q3 = df[column].quantile(0.75)
	IQR = Q3 - Q1

	# Definir los límites para detectar outliers
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR

	# Filtrar el DataFrame para eliminar los outliers
	filtered_df = df[(df[column] >= lower_bound) & 
									(df[column] <= upper_bound)]
	
	return filtered_df

def merge_by_column(df1: pd.DataFrame, df2: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
	"""
	Merge two DataFrames based on a specified column.

	Parameters:
	- df1 (pd.DataFrame): The first DataFrame to merge.
	- df2 (pd.DataFrame): The second DataFrame to merge.
	- column (str): The name of the column to merge on.

	Returns:
	- pd.DataFrame: The merged DataFrame.

	This function merges the two DataFrames based on the specified column. 
	The merged DataFrame contains all the rows from both DataFrames, including rows where the column values are null in either DataFrame.
	"""
	merged_df = pd.merge(df1, df2, left_on=column1, right_on=column2)

	return merged_df

def plot_distribution(df: pd.DataFrame, column: str, filter_outliers: bool = True) -> None:
	"""
	Plots the distribution of a specified column in a DataFrame.

	Parameters:
	- df (pd.DataFrame): The DataFrame containing the column to plot.
	- column (str): The name of the column to plot.
	- filter_outliers (bool): Optional parameter to filter out outliers before plotting (default is True).

	Returns:
	- None
	"""
	if filter_outliers:
		df = get_filtered_outliers(df, column)

	# Crear el gráfico de distribución
	sns.histplot(df[column], kde=True, color='blue')
	plt.title(f"Population Mean: {np.mean(df[column])} \n Population Std Dev: {np.std(df[column])}")
	plt.show()

def plot_CLT(df: pd.DataFrame, column: str, filter_outliers: bool = True, sample_size = 100) -> None:
	"""
	Plots the Central Limit Theorem for a specified column in a DataFrame.

	Parameters:
	- df (pd.DataFrame): The DataFrame containing the column to plot.
	- column (str): The name of the column to plot.
	- filter_outliers (bool): Optional parameter to filter out outliers before plotting (default is True).

	Returns:
	- None
	"""
	if filter_outliers:
		df = get_filtered_outliers(df, column)

	sample_means = []
	for i in range(500):
		samp = np.random.choice(df[column], sample_size, replace=False)
		sample_means.append(samp.mean())

	mean_sampling_distribution = round(np.mean(sample_means),3)
	std_sampling_distribution = round(np.std(sample_means),3)

	sns.histplot(sample_means, stat = 'density')
	# calculate the mean and SE for the probability distribution
	mu = np.mean(df[column])
	sigma = np.std(df[column])/(sample_size**.5)

	# plot the normal distribution with mu=popmean, sd=sd(pop)/sqrt(samp_size) on top
	x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
	plt.plot(x, stats.norm.pdf(x, mu, sigma), color='blue', label = 'normal PDF')
	# plt.axvline(mean_sampling_distribution,color='r',linestyle='dashed')
	plt.title(f"Sampling Dist Mean: {mean_sampling_distribution} \n Sampling Dist Standard Deviation: {std_sampling_distribution}")
	plt.show()

def plot_violinplot(df: pd.DataFrame, column: str, filter_outliers: bool = True) -> None:
	"""
	Plots the violin plot of a specified column in a DataFrame.

	Parameters:
	- df (pd.DataFrame): The DataFrame containing the column to plot.
	- column (str): The name of the column to plot.
	- filter_outliers (bool): Optional parameter to filter out outliers before plotting (default is True).

	Returns:
	- None
	"""
	if filter_outliers:
		df = get_filtered_outliers(df, column)

	# Crear el gráfico de distribución
	sns.violinplot(df[column], color='blue')
	plt.title(f"Population Mean: {np.mean(df[column])} \n Population Std Dev: {np.std(df[column])}")
	plt.show()

def plot_merged_boxplot(df1: pd.DataFrame, df2: pd.DataFrame, x: str, y: str, merge_col1: str, merge_col2: str, hue:str = None, filter_outliers: bool = True) -> None:
	"""
	Plots a merged boxplot of two DataFrames based on a specified column.

	Parameters:
	- df1 (pd.DataFrame): The first DataFrame to merge.
	- df2 (pd.DataFrame): The second DataFrame to merge.
	- x (str): The column name for the x-axis.
	- y (str): The column name for the y-axis.
	- merge_col1 (str): The column name in df1 to merge on.
	- merge_col2 (str): The column name in df2 to merge on.
	- hue (str): Optional parameter to specify the column name for hue (default is None).
	- filter_outliers (bool): Optional parameter to filter out outliers before plotting (default is True).

	Returns:
	- None
	"""

	merged_df = merge_by_column(df1, df2, merge_col1, merge_col2)

	if filter_outliers:
		merged_df = get_filtered_outliers(merged_df, x)

	plt.figure(figsize=(10, 6))
	sns.boxplot(x=x, y=y, data=merged_df, hue=hue)
	plt.show()

def plot_boxplot(x: str, y: str, df: pd.DataFrame, filter_outliers: bool = True, title: str = None) -> None:
	"""
	Plots a boxplot of a specified column in a DataFrame.

	Parameters:
	- x (str): The name of the column to plot on the x-axis.
	- y (str): The name of the column to plot on the y-axis.
	- df (pd.DataFrame): The DataFrame containing the data to plot.
	- filter_outliers (bool): Optional parameter to filter out outliers before plotting (default is True).
	- title (str): Optional parameter to set the title of the plot (default is None).

	Returns:
	- None
	"""
	if filter_outliers:
		df = get_filtered_outliers(df, x)

	# Crear el gráfico de distribución
	sns.boxplot(x=x, y=y, data=df)
	plt.title(title)
	plt.show()