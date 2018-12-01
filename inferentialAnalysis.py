import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

# Run a t-test
def runTTest(ivA, ivB, dv):
    ttest = scipy.stats.ttest_ind(ivA[dv], ivB[dv])
    print(ttest)

# Run ANOVA
def runAnova(data, formula):
    model = ols(formula, data).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print(aov_table)

def getPercentage(df):
    df['sum'] = df['Admitted'] + df['Rejected']
    df['pecentage_admitted'] = df['Admitted']/df['sum']
    df['pecentage_rejected'] = df['Rejected']/df['sum']
    #df.to_csv(file_name, sep='\t', encoding='utf-8')
    return df

# Run the analysis
rawData, df = generateDataset('simpsons_paradox.csv')
new_df = getPercentage(df)
print("Does gender correlate with percentage of admission?")
men = new_df[(new_df['Gender']== 'Male')]
women = new_df[(new_df['Gender'] == 'Female')]
runTTest(men, women, 'pecentage_admitted')

print("Does department correlate with admissions?")
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData, simpleFormula)

print("Do gender and department correlate with admissions?")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

#print("get percentage")
#new_df = getPercentage(df)
##print(new_df.to_string())
#print(new_df[['Gender','Mean_Age','pecentage_admitted','pecentage_rejected','Department']].to_string())
#print(new_df)

##1
#(a).T-test
#(b).Generalized Regression
#(c).T-test
#(d).Chi-Squared Test
#
##2
#The factor of gender relatively appears to contribute most heavily to admission. Since the p-value is closed to 0.1 but it still doesn't have significant relation to admission. I don't think the admission process is biased. First, the gender factor and other factors don't appear to have a significant correlation with Admission. Also, from the data of percentage of admitted, these data didn't appear to have a big difference between gender and department. So I think the admission process is biased.
#
##3
#before corrected
#Does gender correlate with admission?
#Ttest_indResult(statistic=5.332277756733584, pvalue=0.001774285663548817)
#Does department correlate with admissions?
#                  sum_sq   df         F    PR(>F)
#C(Department)   92266.75  5.0  0.737438  0.622205
#Residual       150141.50  6.0       NaN       NaN
#Do gender and department correlate with admissions?
#                      sum_sq   df          F    PR(>F)
#C(Department)   41153.333333  5.0   7.515304  0.036670
#C(Gender)      145760.750000  2.0  66.546025  0.000851
#Residual         4380.750000  4.0        NaN       NaN

#after corrected
#Does gender correlate with admission?
#Ttest_indResult(statistic=1.5797839827086906, pvalue=0.14861367928484936)
#Does department correlate with admissions?
#                  sum_sq   df         F    PR(>F)
#C(Department)   92266.75  5.0  0.737438  0.622205
#Residual       150141.50  6.0       NaN       NaN
#Do gender and department correlate with admissions?
#                      sum_sq   df         F    PR(>F)
#C(Department)   79566.233333  5.0  0.619261  0.697031
#C(Gender)       47352.900000  2.0  0.921365  0.468693
#Residual       102788.600000  4.0       NaN       NaN
#
#the correlation relates "Gender" column is different between old results and new results
#Before corrected, gender factor and department factor all have a strong significant correlation to admission. After corrected, both of the factors don't appear to have a significant correlation to admission.
#
##4
#From the visualization of the number of people admitted by departments, we can see from A department to F department, number of people admitted is decreased. Department A, B and D admitted more men than women while department C, E and F admitted more women than men.
#From the visualization of the numebr of people rejected by departments, we can see from department F rejected the number of people the most. It is interesting to see though department A admitted more men, it rejected more men than women as well.
