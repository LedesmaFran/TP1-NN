import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal


def meanVector_covMatrix(class_vector,df,columns,onlyclass=None,plot=True):
    if onlyclass is not None:
        class_vector=onlyclass
    result = {}
    for class_name in class_vector:
        result_temp = []
        class_data = df[df['Class']==class_name]
        mean_vector = []
        for col in columns:
                mean = class_data[col].mean()
                mean_vector.append(round(mean,4))

        covMatrix = pd.DataFrame.cov(class_data)

        result_temp.append(mean_vector)
        result_temp.append(covMatrix)

        result[class_name] = result_temp

        if plot:
            print("Name class:", class_name)
            print('u_vector =',mean_vector)
            print('cov_matrix:\n',round(covMatrix,4),end='\n\n')
            f = plt.figure(figsize=(8, 6))
            plt.matshow(df.corr(), fignum=f.number)
            plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
            plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Correlation Matrix: '+class_name, fontsize=16);
            plt.show()
    return result

# def meanVector_covMatrix(class_vector,df,columns,onlyclass=None,plot=True):
#     if onlyclass is not None:
#         class_vector=onlyclass
#     result = []
#     for class_name in class_vector:
#         result_temp = []
#         class_data = df[df['Class']==class_name]
#         mean_vector = []
#         for col in columns:
#                 mean = class_data[col].mean()
#                 mean_vector.append(round(mean,4))

#         covMatrix = pd.DataFrame.cov(class_data)

#         result_temp.append(mean_vector)
#         result_temp.append(covMatrix)

#         if plot:
#             print("Name class:", class_name)
#             print('u_vector =',mean_vector)
#             print('cov_matrix:\n',round(covMatrix,4),end='\n\n')
#             f = plt.figure(figsize=(8, 6))
#             plt.matshow(df.corr(), fignum=f.number)
#             plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
#             plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
#             cb = plt.colorbar()
#             cb.ax.tick_params(labelsize=14)
#             plt.title('Correlation Matrix: '+class_name, fontsize=16);
#             plt.show()
#         result.append(result_temp)
#     return result



def plot_data_as_gaussian(class_vector,df,columns,onlyclass=None):
    if onlyclass is not None:
        class_vector=onlyclass
    for class_name in class_vector:
        class_data = df[df['Class']==class_name]
        plt.figure(figsize=(8,6),dpi=100)
        for col in columns:
            mean = class_data[col].mean()
            std = class_data[col].std()
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            plt.plot(x, stats.norm.pdf(x, mean, std),label=col+' $\mu=$'+str(round(mean,3))+' $\sigma=$'+str(round(std,3)))
            plt.legend()
            
        plt.title(class_name)
        plt.grid()
        plt.show()



def plot_data(class_vector,df,columns,onlyclass=None):
    if onlyclass is not None:
        class_vector=onlyclass
    for class_name in class_vector:
        class_data = df[df['Class']==class_name]
        plt.figure(figsize=(8,6),dpi=100)
        for col in columns:
            mean = class_data[col].mean()
            std = class_data[col].std()
            plt.hist(class_data[col], bins=20,label=col+' $\mu=$'+str(round(mean,3))+' $\sigma=$'+str(round(std,3)))
            plt.legend()
            plt.gca().set(title=class_name, ylabel='Repeticiones')
        plt.grid()
        plt.show()
        print("\n")

def print_data(class_vector,df,columns,onlyclass=None):
    if onlyclass is not None:
        class_vector=onlyclass
    for i in class_vector:
        class_data = df[df['Class']==i]
        print("Name class:", i)
        for col in columns:
            print(col,end=' ') 
            print("mean:",round(class_data[col].mean(),3), end='\t')
            print("desv std:",round(class_data[col].std(),3))
        print("\n")
