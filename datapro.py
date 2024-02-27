import pandas as pd
import numpy as np
#import statsmodels.stats.api as sm

import statsmodels.api as sm

import statsmodels.stats.api as sms

from statsmodels.stats.outliers_influence import variance_inflation_factor
#from matplotlib import pyplot
import seaborn as sns
from matplotlib import pyplot as plt

def bp_test(res):
    """
    Returns a data frame with the Breusch-Pagan test
    
    Parameters
    -----------
    res: The result of a smf.ols.fit() oject.
    """
    name = ["Lagrange multiplier LM statistic", "LM p-value", "F-value", "Fp-value"]
    test = sms.het_breuschpagan(res.resid, res.model.exog)
    return pd.DataFrame(test,index=name,columns=["Breusch-Pagan test"])

def feasible_gls(data,res):
    """
    Feasible Generalized Least Squares

    Parameters
    ----------
    data: The data frame that includes the data of the independent variables
    res: The result of a smf.ols.fit() oject.
    """
    lres_u=np.log(pow(res.resid,2))
    data["lres_u"]=lres_u
    
    varn=list(pd.DataFrame(res.params).index) 
    varn[0]="const"
    X1=data[varn]
    y1=data["lres_u"]
    mod=sm.OLS(y1,X1)
    res_ols1 = mod.fit()
    
    hhat=np.exp(list(res_ols1.fittedvalues))
    X=data[varn]
    y=res.model.endog
    gls_model = sm.GLS(y, X, sigma=hhat)
    gls_results = gls_model.fit()
    return print(gls_results.summary())
    
def plot_fit(res,x,y,reg_line=True):
    """
    Plot of a OLS whit the regression line, for a simple regression model
    
    Parameters
    -----------
    res: The result of a smf.ols.fit() oject for a simple regression model.
    x: Data frame with X´s variables
    y: Pandas Serie with the dependent variable 
    """
    if reg_line==True:
        alpha=res.params[0]
        beta=res.params[1]
        ypred = alpha + beta * x
        plt.figure(figsize=(10, 4))
        plt.plot(x ,ypred)     # regression line
        plt.plot(x, y, 'ro')   # scatter plot showing actual data
        plt.title('OLS Regression Line')
        plt.xlabel(x.keys()[1])
        plt.ylabel(y.name)
    else:
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, 'ro')   # scatter plot showing actual data
        plt.xlabel(x.keys()[1])
        plt.ylabel(y.name)
    plt.show()

def robust_se (res):
    """
    Returns a data frame with the coeficients of the regression
    Parameters
    -----------
    res: The result of a smf.ols.fit() oject.
    """
    df=pd.DataFrame(res.params,columns=["coef"])
    df["std err"]=res.bse
    df["HC0 std err"]=res.HC0_se
    df["HC1 std err"]=res.HC1_se
    return df


def vif(data,res,cons=False):
    """
    Estimates the variance_inflation_factor from a OLS model. Returns a data frame with the VIF factor in the first 
    column, ordered ascending, and the variable name in the second one.

    Parameters
    -----------
    data: The data frame that includes the data of the independent variables
    res: The result of a smf.ols.fit() oject.
    cons: Bolean, default False
        The behavior is as follows:
        bool. If True -> Constant appears in the resulting data frame
    
    """
    varn=list(pd.DataFrame(res.params).index)
    varn[0]="const"
    X=data[varn]
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["Variable"] = X.columns
    if cons==False:
        vif=vif.iloc[1:,]
    return vif

def corr_matrix(data,res):
    """
    Correlation matrix using heatmaps of the seabonr module
    Parameters
    -----------
    data: The data frame that includes the data of the independent variables
    res: The result of a smf.ols.fit() oject.
    """
    varn=data[list(pd.DataFrame(res.params).index)[1:]]
    correlation = varn.corr()
    plt.figure(figsize=(10,10))
    plt.title('Correlation Matrix')
    return sns.heatmap(correlation, vmin=-1, vmax=1,  square=True,annot=True,cmap='cubehelix')


