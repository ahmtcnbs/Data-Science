import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# hypothesis testing
from scipy.stats import shapiro
import scipy.stats as stats

# configuration
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore',category=FutureWarning)

pd.set_option('display.max_columns',None)
pd.options.display.float_format = '{:.4f}'.format

# DATA OPERATIONS

df = pd.read_csv('cookie_cats.csv')
print(df.head())
# Number of Unique User
print(df.userid.nunique() == df.shape[0])
# Summary Stats: sum_gamerounds
print(df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]].T)

print(df.groupby('version').sum_gamerounds.agg(['count', 'median', 'mean', 'std', 'max']))

df = df[df.sum_gamerounds < df.sum_gamerounds.max()]
print(df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]].T)

fig = plt.figure(figsize=(7, 7))
sns.displot(data=df, x='sum_gamerounds', kde=True, label='Sum Gamerounds firts 14 days.')
plt.show()

# sns.countplot(data=df,x='retention_1')
# sns.countplot(data=df,x='retention_7')
plt.show()

print(df.groupby("sum_gamerounds").userid.count().reset_index().head(20))
# how many users reached gate 30 & gate 40 levels ?

print(df.groupby("sum_gamerounds").userid.count().loc[[30,40]])

# a/b groups & target summary stats
print(df.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"]))

print(df.groupby(["version", "retention_1"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]))

print(df.groupby(["version", "retention_7"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]))

df['Retention'] = np.where((df.retention_1 == True) & (df.retention_7 == True),1,0)
print(df.groupby(['version', 'Retention'])['sum_gamerounds'].agg(['count', 'median', 'mean', 'std', 'max']))

df["NewRetention"] = list(map(lambda x,y: str(x)+"-"+str(y), df.retention_1, df.retention_7))
print(df.groupby(["version", "NewRetention"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]).reset_index())

df['version'] = np.where(df.version == "gate_30","A","B")
print(df.head())

# A/B Testing Function

def AB_Test(dataframe, group, target):
    # Split A/B
    groupA = dataframe[dataframe[group] == "A"][target]
    groupB = dataframe[dataframe[group] == "B"][target]

    # Assumption: Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05

    # H0: Distribution is Normal - False
    # H1: Distribution is not Normal - True

    if (ntA == False) & (ntB == False): # H0: Normal Distribution
        # parametric test
        # assumption: Homogeneity of Variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0: Homogeneity: False
        # H1: Heterogeneous: True
        if leveneTest == False:
            # homogeneity
            t_test = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            t_test = stats.ttest_ind(groupA, groupB, equal_var=False)
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        t_test = stats.mannwhitneyu(groupA, groupB)[1]
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True

    # RESULT
    temp = pd.DataFrame({
        'AB Hypothesis':[t_test < 0.05],
        'p-value':[t_test]
    })

    temp['Test Type'] = np.where((ntA == False) & (ntB == False), "Parametric",'Non-Parametric')

    temp['AB Hypothesis'] = np.where(temp['AB Hypothesis'] == False, 'Fail to Reject H0','Reject H0')

    temp['Comment'] = np.where(temp['AB Hypothesis'] == "Fail to Reject H0",'A/B Groups are similar','A/B Groups are not similar.')

    # Columns

    if(ntA == False) & (ntB == False):
        temp['Homogeneity'] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[['Test Type','Homogeneity','AB Hypothesis','p-value','Comment']]
    else:
        temp = temp[['Test Type','AB Hypothesis','p-value','Comment']]

    # Print Hypothesis

    print("A/B Testing Hypothesis")
    print("H0: A == B")
    print('H1: A != B', "\n")
    return temp

AB_Test(dataframe=df,group="version", target="sum_gamerounds")
print(AB_Test(dataframe=df,group="version", target="sum_gamerounds"))

a = df.groupby('version').retention_1.mean()
b = df.groupby('version').retention_7.mean()
print(a,b)




















