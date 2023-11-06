import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # the KNN model
from sklearn.metrics import f1_score

import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce

titanic_variance_based_split = 107
customer_variance_based_split = 113 

class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

#This class will rename one or more columns.
class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self


  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
   # assert self.mapping_dict in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_dict}"'

    keys_values = self.mapping_dict.keys()

    column_set = set(X.columns)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    assert not keys_not_found, f'{self.__class__.__name__}[{self.mapping_dict}] these mapping keys do not appear in the column: {keys_not_found}\n'

    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result


class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    assert isinstance(target_column, str), f'Error: {self.__class__.__name__} constructor expected a string for target_column but got {type(target_column)} instead.'
    assert isinstance(dummy_na, bool), f'Error: {self.__class__.__name__} constructor expected a boolean for dummy_na but got {type(dummy_na)} instead.'
    assert isinstance(drop_first, bool), f'Error: {self.__class__.__name__} constructor expected a boolean for drop_first but got {type(drop_first)} instead.'
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y=None):
    print(f"\nWarning: The {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'Error: {self.__class__.__name__}.transform expected a DataFrame but got {type(X)} instead.'
    assert  self.target_column in X.columns.to_list(), f' {self.__class__.__name__} transform unknown column "{self.target_column}"'
    X_ = pd.get_dummies(X, columns=[self.target_column], dummy_na=self.dummy_na, drop_first=self.drop_first)
    return X_

  def fit_transform(self, X, y=None):
    print(f'Warning: {self.__class__.__name__}.fit_transform is not supported. Please use transform() instead.')
    return self.transform(X)


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.high = None
    self.low = None
    self.fitt= False

  def fit(self,X, y=None):
    assert isinstance(X, pd.core.frame.DataFrame), f'Expected DataFrame but got {type(X)} instead.'
    column_data = X[self.target_column]
    self.high = column_data.mean()+3 *column_data.std()
    self.low = column_data.mean()-3 *column_data.std()
    self.fitt= True
    return self

  def transform(self,X):
    assert self.fitt, f'NotFittedError:Call fit before transform'
    X_=X.copy()
    X_[self.target_column]= X[self.target_column].clip(upper=self.high)
    return X_

  def fit_transform(self,X):
    self.fit(X)
    return self.transform(X)


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']

    self.fence= fence
    self.fence_l= None
    self.fence_h=None
    self. target_column=target_column
    self.fitt= False
    

  def fit(self,X,y=None):
    '''q1 = transformed_df[column].quantile(0.25)
        q3 = transformed_df[column].quantile(0.75)
        iqr = q3-q1
        outer_low = q1-3*iqr
        outer_high = q3+3*iqr'''
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'

    q1 = X[self.target_column].quantile(0.25)
    q3 = X[self.target_column].quantile(0.75)
    iqr = q3-q1
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    inner_low = q1-1.5*iqr
    inner_high = q3+1.5*iqr
    
    if self.fence=="outer":
      self.fence_l= outer_low
      self.fence_h=outer_high
      self.fitt=True
    elif self.fence=="inner":
      self.fence_l= inner_low
      self.fence_h=inner_high
      self.fitt=True
    
    
    return self

  def transform(self,X):
    assert self.fitt, f'NotFittedError:Call fit before transform'
    X_=X.copy()
    X_[self.target_column]= X[self.target_column].clip(lower=self.fence_l,upper=self.fence_h)
    X_.reset_index(inplace=True,drop=True)
    return X_

  def fit_transform(self,X,y=None):
    self.fit(X,y)
    return self.transform(X)

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    #fill in rest below
    self.column= column
    self.med = None
    self.iqr = None
    self.fitt = False

  def fit(self,X,y=None):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    self.med = X[self.column].median()
    self.iqr= X[self.column].quantile(0.75) - X[self.column].quantile(0.25)
    self.fitt = True
    return self

  def transform(self,X):
    assert self.fitt, f'NotFittedError:Call fit before transform'
    new = X.copy()
    new[self.column] -= self.med
    new[self.column] /= self.iqr
    return new

  def fit_transform(self, X,y=None):
    self.fit(X,y)
    return self.transform(X)

def find_random_state(features_df, labels, n=200):
  var=[]
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                            random_state=i, stratify=labels)
    model.fit(train_X, train_y)  # train model
    train_pred = model.predict(train_X)  # predict against training set
    test_pred = model.predict(test_X)  # predict against test set
    train_f1 = f1_score(train_y, train_pred)  # F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)  # F1 on test predictions
    f1_ratio = test_f1 / train_f1  # take the ratio
    var.append(f1_ratio)

  rs_value = sum(var) / len(var)  # get average ratio value
  indices = np.array(abs(var - rs_value)).argmin()  # find the index of the smallest value
  return indices

titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  #from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  #from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer('Age')), #from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')), #from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
    ], verbose=True)

def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below
  labels = original_table[label_column_name].to_list()
  feat = original_table.drop(columns = label_column_name)
  X_train, X_test, y_train, y_test = train_test_split(feat, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train, y_train)
  X_test_transformed = the_transformer.transform(X_test)

  X_train_numpy = X_train_transformed.to_numpy()
  X_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test) 
  return X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy
  
def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs=rs, ts=ts)
  
def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs=rs, ts=ts)
