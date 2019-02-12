#!/usr/bin/python
# -*- coding: utf-8 -*-
# Script to create an SARIMAX model and make a forecast

import statsmodels.api as sm
import itertools
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class Files:
	
	def get_series(self):
		self.data_exo = pd.read_csv('exo.csv')
		self.data_main = pd.read_csv('main.csv')

	def save_model(self):
		with open('arima.dat', 'wb') as pkl:
			pickle.dump(self.arima_model, pkl)
			#pickle.dump(self.lambda_boxcox, pkl)

	def load_model(self):
		with open('arima.dat', 'rb') as pkl:
			self.arima_model = pickle.load(pkl)
			#self.lambda_boxcox = pickle.load(pkl)
		return(self.arima_model)

	def save_forecast(self):
		self.df_final.to_csv('forecast_mean.csv')

class Model:

	def select_data(self):
		# Merge columns into a single dataframe of observed values, based on date
		dataset = files.data_main.join(files.data_exo.set_index('date'), on='date').dropna()
		# Select part of the precipitation dataframe that corresponds to the forecast
		obs_end = dataset.tail(1)['date'].values[0]
		exo_prev = files.data_exo[(files.data_exo['date'] > obs_end)]
		# Select predict dates
		self.dates_prev = exo_prev['date']
		# Reshape
		endo_obs = np.array(dataset['endo_value'])
		self.endo_obs = endo_obs.reshape(-1, 1)
		exo_obs = np.array(dataset['exo_value'])
		self.exo_obs = exo_obs.reshape(-1, 1)
		exo_prev = np.array(exo_prev['exo_value'])
		self.exo_prev = exo_prev.reshape(-1, 1)
	
	def normalize(self):
		# Calculate lambda only if doesn't have zero values
		n_zeros = len(self.endo_obs[self.endo_obs <= 0])
		if n_zeros == 0:
			self.endo_obs2, self.lambda_boxcox = boxcox(self.endo_obs)
		else:
			self.lambda_boxcox = -999
		# Limit lambda values
		if abs(self.lambda_boxcox[0]) > 1:
			self.endo_obs2 = self.endo_obs
			self.lambda_boxcox = -999
		#print(self.endo_obs2, self.lambda_boxcox)
	
	def run_auto(self):
		self.arima_model = auto_arima(self.endo_obs2, start_p=0, start_d=0, start_q=0, max_p=3, max_d=1, max_q=3,
                          start_P=0, start_Q=0, D=1, seasonal=False, m=1,
                          exogeneous=self.exo_obs, 
                          trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
		#print(model.arima_model.summary())
		# Compile parameters to list
		self.parameters = [self.arima_model.order,self.arima_model.seasonal_order,self.lambda_boxcox[0],self.arima_model.aic()]
		print(self.parameters)
		return(self.arima_model)
		
	def run_auto_arimax(self):
		lower_aic = float(99999)
		best_pdq = [0,0,0]
		param = list(itertools.product(range(0,4), range(0,2), range(0,4)))
		for pdq in param:
			#print(pdq)
			try:
				self.arima_model = ARIMA(order=pdq, suppress_warnings=True).fit(y=self.endo_obs2, exogenous=self.exo_obs)
				if self.arima_model.aic() < lower_aic:
					lower_aic = self.arima_model.aic()
					best_pdq = tuple(self.arima_model.order)
			except:
				continue
		#print(model.arima_model.summary())
		# Compile parameters to list
		self.parameters = [best_pdq,self.lambda_boxcox[0],lower_aic]
		print(self.parameters)
		return(self.arima_model)

	def run_auto_sarimax(self):
		lower_aic = float(99999)
		best_pdq = [0,0,0]
		best_spdq = [0,0,0]
		param = list(itertools.product(range(0,4), range(0,2), range(0,4)))
		m = 1 # frequency
		param_seasonal = [(x[0], x[1], x[2], m) for x in list(itertools.product(range(0,4), range(0,2), range(0,4)))]
		for pdq in param:
			for spdq in param_seasonal:
				try:
					mod = sm.tsa.statespace.SARIMAX(self.endo_obs2, exog=self.exo_obs, order=pdq, seasonal_order=spdq, enforce_stationarity=False, enforce_invertibility=False)
					self.arima_model = mod.fit(disp=0)
					print('ARIMA{}x{}{} - AIC:{}'.format(pdq, spdq, m, self.arima_model.aic))
					if self.arima_model.aic() < lower_aic:
						lower_aic = self.arima_model.aic()
						best_pdq = tuple(pdq)
						best_spdq = tuple(spdq)
				except:
					continue
		#print(model.arima_model.summary())
		# Compile parameters to list
		self.parameters = [best_pdq,best_spdq,self.lambda_boxcox[0],lower_aic]
		print(self.parameters)
		return(self.arima_model)

	def forecast(self):
		#self.predict = self.arima_model.predict(n_periods=self.exo_prev.shape[0], exogenous=self.exo_prev)
		self.predict = self.arima_model.predict(n_periods=self.exo_prev.shape[0], exogenous=self.exo_prev, return_conf_int=True, alpha=0.7)
		
	def renormalize(self):
		if self.lambda_boxcox == float(-999):
			self.predict_mean = self.predict[0]
			self.predict_down = self.predict[1][:,0]
			self.predict_up = self.predict[1][:,1]
		else:
			self.predict_mean = inv_boxcox(self.predict[0], self.lambda_boxcox)
			self.predict_down = inv_boxcox(self.predict[1][:,0], self.lambda_boxcox)
			self.predict_up = inv_boxcox(self.predict[1][:,1], self.lambda_boxcox)
		# Join predict dates with values into a dataframe
		df_final = pd.DataFrame(self.predict_mean, self.dates_prev)
		df_final.columns = ['endo_value']
		return(df_final)

# Crete objects
files = Files()
model = Model()

# Get main and exogeneous time series
files.get_series()

# Select observed and predict data + reshape
model.select_data()

# Normalize serie
model.normalize()

# Run auto_arima to save parameters/model
files.arima_model = model.run_auto_arimax()

# Save model
files.save_model()

# Load model
model.arima_model = files.load_model()

# Make predictions
model.forecast()

# Renormalize
files.df_final = model.renormalize()

# Save predictions do CSV
files.save_forecast()
