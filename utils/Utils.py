import numpy as np
import os, glob
import re
import pandas as pd
import json

dirname = os.path.dirname(__file__)

class Utils:
	def __init__(self):
		self.turboengine_idx = 1
		self.bearing_idx = 4

	def get_features(self, component_type):

		def sorted_aphanumeric(data):
		    convert = lambda text: int(text) if text.isdigit() else text.lower()
		    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		    return sorted(data, key=alphanum_key)

		def state_of_health(data):
			initial_capacity = data[0]
			for i in range(len(data)):
				data[i] = (data[i]/initial_capacity)*100
			return data

		def get_battery_data():
			data_dir='examples/data/battery_data/B0006_discharge.json'
			data_dir = dirname + '/' + data_dir
			with open(data_dir) as f:    
			    discharge_data = json.load(f)
			data = []
			time_stamps = []
			for cycle in discharge_data.keys():
				data.append(discharge_data[cycle]["capacity"][0]) # remove the last dimension
				time_stamps.append(discharge_data[cycle]["date_time"])
			soh = np.array(state_of_health(data)).reshape(len(data), 1)
			times = np.array(time_stamps).reshape(len(time_stamps), 1)
			return soh

		if (component_type == 'turboengine'):
			data_dir='examples/data/phm08_data/csv'
			data_dir = dirname + '/' + data_dir
			dirlist = sorted_aphanumeric(os.listdir(data_dir))
			all_data = []
			for file in dirlist:
				if (not file.endswith('.py')) and (file != '__pycache__'):
					data = pd.read_csv(data_dir+"/"+file, engine='python')
					data = np.array(data)
					all_data.append(data[:, 2:])
			all_data_np = np.array(all_data)
			return all_data_np[self.turboengine_idx]
		elif (component_type == 'bearing'):
			data_dir='examples/data/bearing_data/'
			data_dir = dirname + '/' + data_dir
			rms = np.load(data_dir+'/3rd_test_bearing_'+str(self.bearing_idx)+'_rms.npy', allow_pickle=True)
			kurtosis = np.load(data_dir+'/3rd_test_bearing_'+str(self.bearing_idx)+'_kurtosis.npy', allow_pickle=True)
			crest = np.load(data_dir+'/3rd_test_bearing_'+str(self.bearing_idx)+'_crest.npy', allow_pickle=True)
			all_data_np = np.concatenate((rms, kurtosis, crest), axis=1)
			return all_data_np

		elif (component_type == 'battery'):
			return get_battery_data()
			
	def check1(self, data):
		assert isinstance(data,np.ndarray),'*** Please provide numpy array as input ***'
		assert len(data.shape)>0,'*** Ensure the array is not empty ***'
		assert np.all(np.isfinite(data)),'*** Remove NaN and Inf values from the numpy array'

	# Count the number of positive peaks
	def count_pos(self, data):
	    if(len(data.shape) == 1):
	        count = (data>0).sum()
	    else:
	        count = [(column>0).sum() for column in data.T]
	    return np.asarray(count)

	def count_neg(self, data):
	    if(len(data.shape) == 1):
	        count = (data<0).sum()
	    else:
	        count = [(column<0).sum() for column in data.T]
	    return np.asarray(count)


	def zscore(self, data, mu, sigma):
		z_s = (data - mu)/sigma
		return z_s    

	def zs_param(self, train, percentileScore=None):
		if(percentileScore == None):
			mu, sigma = np.mean(train), np.std(train)
		else:
			mu, sigma = np.percentile(train,percentileScore), np.std(train)
		return mu, sigma

	def get_strangeness(self, data, normal_mean):
		return (data - normal_mean)

import numpy as np, pandas as pd, pathlib as pl, os
from collections import namedtuple
# ===========================================
DeviationContext = namedtuple('DeviationContext', 'strangeness pvalue deviation is_deviating')


# ===========================================
class NotFittedError(Exception): pass
class InputValidationError(Exception): pass
class TestUnitError(Exception): pass
class NoRefGroupError(Exception): pass


# ===========================================
def append_to_df(df, dt, x): # Appends a new row to a DataFrame
    if df is None or len(df) == 0:
        if len(x) > 0: return pd.DataFrame( data = [x], index = [dt] )
        else: return pd.DataFrame( data = [], index = [] )
    else:
        if len(x) > 0: df.loc[dt] = x
        return df


# ===========================================
def create_directory_from_path(pathname):
    pathname = pl.Path(pathname).resolve()
    directory = os.path.dirname(os.path.abspath(pathname))
    if not os.path.exists(directory):
        os.makedirs(directory)
    return pathname

# ===========================================
# TODO add week-end
def dt2num(dt, criterion):
        if criterion == "hour-of-day":
            return dt.hour
        elif criterion == "day-of-week":
            return dt.weekday()
        elif criterion == "day-of-month":
            return dt.day
        elif criterion == "week-of-year":
            return dt.isocalendar()[1]
        elif criterion == "month-of-year":
            return dt.month
        elif criterion == "season-of-year":
            season = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
            return season[dt.month]
        else:
            raise InputValidationError("Unknown criterion {} in ref_group.".format(criterion))


# ===========================================
def validate_measure_str(string):
    strings = ["median", "knn", "lof"]
    if string not in strings:
        raise InputValidationError("measure should be one of {}. Given measure = '{}'".format(strings, string))


def validate_int_higher(k, low):
    if not k > low: raise InputValidationError("The specified integer {} should be higher than {}".format(k, low))


def validate_list_not_empty(lst):
    if len(lst) == 0:
        raise InputValidationError("The list should not be empty")


def validate_all_in_range(lst, bounds):
    low, up = bounds
    for v in lst:
        if not (low <= v <= up):
            raise InputValidationError("Values of the list should be in [{}, {}]".format(low, up))


def validate_is_fitted(is_fitted):
    if not is_fitted:
        raise NotFittedError("'fit' must be called before calling 'get'")


def validate_reference_group(Xref):
    if len(Xref) == 0:
        raise NoRefGroupError("Empty reference group data.")


def validate_reference_grouping_input(uid_test, dt, dffs):
    if not (0 <= uid_test < len(dffs)):
        raise InputValidationError("uid_test should be in range(nb_units). Given uid_test = {}, nb_units is {}".format(uid_test, len(dffs)))
        
    if len(dffs[uid_test]) == 0:
        raise TestUnitError("Test unit (uid={}) does not have data".format(uid_test))
    
    try:
        x = dffs[uid_test].loc[dt].values
    except KeyError:
        raise TestUnitError("Test unit (uid={}) does not have data at time {}".format(uid_test, dt))


def validate_individual_deviation_params(w_martingale, non_conformity, k, dev_threshold, ref_group=None):
    if w_martingale < 1:
        raise InputValidationError("w_martingale should be an integer higher than 0. Given w_martingale = {}".format(w_martingale))
    
    strings = ["median", "knn", "lof"]
    if non_conformity not in strings:
        raise InputValidationError("non_conformity should be one of {}. Given non_conformity = '{}'".format(strings, non_conformity))
    
    if non_conformity == "knn":
        if k < 1:
            raise InputValidationError("k should be an integer higher than 0. Given k = {}".format(k))
    
    if not (0 <= dev_threshold <= 1):
        raise InputValidationError("dev_threshold should be in [0, 1]. Given dev_threshold = {}".format(dev_threshold))

    strings = ["hour-of-day", "day-of-week", "day-of-month", "week-of-year", "month-of-year", "season-of-year"]
    if (ref_group is not None) and (ref_group != "external") and (not isinstance(ref_group, (list, np.ndarray))) and (not callable(ref_group)):
        raise InputValidationError("ref_group should be either a list containing one or many of {}, or the string 'external', "
                                   "or a callable function(times, values, x)"
                                   "Given ref_group = '{}'".format(strings, ref_group))

        
def validate_fit_input(X):
    if len(X) == 0 or not isinstance(X, (list, np.ndarray)):
        raise InputValidationError("X must be a non empty array-like of shape (n_samples, n_features)")


def validate_get_input(x):
    if len(x) == 0 or not isinstance(x,(list, np.ndarray)):
        raise InputValidationError("x must be a non empty array-like of shape (n_features,)")

