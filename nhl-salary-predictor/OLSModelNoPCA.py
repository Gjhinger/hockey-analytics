import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from tensorflow import metrics
# from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import statsmodels.api as sm

# Set up our data

PATH    = "/Users/joshjhinger/Desktop/hockey-analytics/nhl-salary-predictor/"
df = pd.read_csv(PATH + 'data.csv', engine='python')

df['Born'] = pd.to_datetime(df['Born'])
df = df.loc[(df['Born']<='1993')]


# Check how many players we have in our df
print(df)

# Convert "Born" into an age column in our df,
# subtract three because data three years old
now = pd.Timestamp('now')
df['age'] = (now - df['Born']).astype('<m8[Y]') - 3   # 3
print(df)

# Drop columns that we know we don't want to use as predictor variables
del df['City']
del df['Pr/St']
del df['Cntry']
del df['Nat']
del df['Hand']
del df['Last Name']
del df['First Name']
del df['Position'] # this one could be useful
del df['Team']
del df['Born'] # this one could be useful
del df['DftYr']
del df['Ovrl']

# Make undrafted players become drafted in the "ninth" round
df.fillna(9, inplace=True)

# Plot histogram to see skewness of target variable
plt.ticklabel_format(style = 'plain')
plt.hist(df['Salary'])
plt.show()

# Plot a heatmap
plt.figure(figsize=(30,30))

heatmap = sns.heatmap(df.corr(),
            vmin=-1,
            cmap='coolwarm')

plt.show()

# Remove predictive variables b/c of heatmap
del df['Ht']
del df['Wt']
del df['+/-']
del df['E+/-']
del df['PIM']
del df['IPP%']
del df['SV%']
del df['PDO']
del df['A/60']
del df['Pct%']
del df['Diff']
del df['Diff/60']
del df['sDist']
del df['sDist.1']
del df['iHF']
del df['iHF.1']
del df['iHA']
del df['iHDf']
del df['iTKA.1']
del df['iBLK.1']
del df['BLK%']
del df['iFOW']
del df['iFOL']
del df['iFOW.1']
del df['iFOL.1']
del df['FO%']
del df['%FOT']
del df['dzFOW']
del df['dzFOL']
del df['nzFOW']
del df['nzFOL']
del df['ozFOW']
del df['ozFOL']
del df['FOW.Up']
del df['FOL.Up']
del df['FOW.Down']
del df['FOL.Down']
del df['FOW.Close']
del df['FOL.Close']
del df['OTG']
del df['PSG']
del df['PSA']
del df['G.Bkhd']
del df['G.Dflct']
del df['G.Slap']
del df['G.Snap']
del df['G.Tip']
del df['G.Wrap']
del df['CBar ']
del df['S.Bkhd']
del df['S.Dflct']
del df['S.Slap']
del df['S.Snap']
del df['S.Tip']
del df['S.Wrap']
del df['iPenT']
del df['iPenD']
del df['iPenDf']
del df['NPD']
del df['Min']
del df['Maj']
del df['Match']
del df['Misc']
del df['Game']
del df['HF']
del df['HA']
del df['DPS']
del df['OTOI']
del df['Grit']
del df['DAP']
del df['age']
del df['Pace']
del df['RBA']
del df['iPEND']
del df['iPENT']
del df['Post']
del df['G.Wrst']
del df['ENG']
del df['GWG']
del df['1G']
del df['iBLK']
del df['iRB']
del df['iSCF']
del df['iRS']
del df['iDS']
del df['F/60']
del df['SH%']


#Redraw Heatmap
plt.figure(figsize=(20,20))
heatmap2 = sns.heatmap(df.corr(),
            vmin=-1,
            cmap='coolwarm')
plt.show()

print(df.head())

# Principal Component Analysis

y = df.pop('Salary')
X = df

sc = StandardScaler()
X_std = sc.fit_transform(X)

cov_matrix = np.cov(X_std.T)
print('Covariance Matrix \n%s', cov_matrix)

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigen Vectors \n%s', eig_vecs)
print('\n Eigen Values \n%s', eig_vals)
tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)

# Machine Learning

#y = df.pop('Salary')
#X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Create principal components. (Use only 4 significant eigen vectors)
# sklearn_pca = sklearnPCA(n_components=4)
#
# # Transform the data.
# X_train = sklearn_pca.fit_transform(X_train)
#
# # Transform test data.
# X_test = sklearn_pca.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)


# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
#