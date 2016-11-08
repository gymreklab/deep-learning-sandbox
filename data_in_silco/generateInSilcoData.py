'''
This file is used for generating samples which:
1 embeded with specific patterns
2 embeded with partial patterns
3 embeded with no patterns at all
Set 2 and 3 are used as control group

parameters:
see # settings


# work notes
# Experiment 1: 
only generate set 1 and the corresponding labels

# Experiment 2:
only generate set 1 but encode genome into 4-mers

# Experiment 3:
generate set 1, and assign (+) and (-) to each sequence
with the probability:
(equation1)
Prob (+) = 1 when distance <= 30
         = 0 when distance > 30
(equation2)
Prob (+) = 1 when distance <= 30
         = 1/(distance when-30) when distance > 30

# Experiment 4:
One hot version of experiment 3
'''

import tools
from optparse import OptionParser
import random

def generateData_v1():
	print ("Start to generate in silco data")
	print ("====START=======================================")
	##########################
	# read options and args
	###########################
	usage = "usage: %prog [options] <output_file_path>"
	parser = OptionParser(usage)

	# settings
	parser.add_option('--patterns', dest = 'patterns', default='GATA,CAGATG', help='The patterns that are embeded in the positive cases. Separate using comma')
	parser.add_option('--numOfTrainSamples', dest='num_of_train_samples', default=1000, type='int', help='The number of samples used for training [Default: %default]')
	parser.add_option('--numOfTestSamples', dest='num_of_test_samples', default=1000, type='int', help='The number of samples used for testing [Default: %default]')
	parser.add_option('--lenOfSample', dest='len_of_sample', default=50, type='int', help='The length of samples [Default: %default]')

	(options, args) = parser.parse_args()

	###########################
	# Default settings
	###########################
	exp_index = 1
	pattern_arr = options.patterns.split(",")
	pattern_arr_str = "_".join(pattern_arr)
	num_of_train_samples = options.num_of_train_samples
	num_of_test_samples = options.num_of_test_samples
	len_of_sample = options.len_of_sample

	# address for saving
	save_train_data_filename = "EXP_%d/train_data_EXP%d_%s_%d.data" %(exp_index, exp_index, pattern_arr_str, num_of_train_samples)
	save_train_label_filename = "EXP_%d/train_label_EXP%d_%s_%d.label" %(exp_index, exp_index, pattern_arr_str, num_of_train_samples)
	save_test_data_filename = "EXP_%d/test_data_EXP%d_%s_%d.data" %(exp_index, exp_index, pattern_arr_str, num_of_test_samples)
	save_test_label_filename = "EXP_%d/test_label_EXP%d_%s_%d.label" %(exp_index, exp_index, pattern_arr_str, num_of_test_samples)

	##########################
	# Print settings
	##########################
	print ("Number of train samples: %d" %(num_of_train_samples))
	print ("Number of test samples: %d" %(num_of_test_samples))
	print ("Length of each sample: %d" %(len_of_sample))
	print ("Embeded patterns %s" %(pattern_arr_str))
	print ("Train data are saved in %s" %(save_train_data_filename))
	print ("Train label are saved in %s" %(save_train_label_filename))
	print ("Test data are saved in %s" %(save_test_data_filename))
	print ("Test label are saved in %s" %(save_test_label_filename))
	print ("=======================================================")

	##########################
	# Generate Data
	#########################
	tools.dataGenerator(save_train_data_filename, save_train_label_filename, num_of_train_samples, len_of_sample, pattern_arr)
	tools.dataGenerator(save_test_data_filename, save_test_label_filename, num_of_test_samples, len_of_sample, pattern_arr)

	print ("Finish the data generating.")
###########################################################################

def generateData_v2():
	print ("Start to generate in silco data")
	print ("====START=======================================")
	##########################
	# read options and args
	###########################
	usage = "usage: %prog [options] <output_file_path>"
	parser = OptionParser(usage)

	# settings
	parser.add_option('--patterns', dest = 'patterns', default='GATA,CAGATG', help='The patterns that are embeded in the positive cases. Separate using comma')
	parser.add_option('--numOfTrainSamples', dest='num_of_train_samples', default=1000, type='int', help='The number of samples used for training [Default: %default]')
	parser.add_option('--numOfTestSamples', dest='num_of_test_samples', default=1000, type='int', help='The number of samples used for testing [Default: %default]')
	parser.add_option('--lenOfSample', dest='len_of_sample', default=25, type='int', help='The length of samples [Default: %default]')

	(options, args) = parser.parse_args()

	###########################
	# Default settings
	###########################
	exp_index = 1
	kmer = 4
	pattern_arr = options.patterns.split(",")
	pattern_arr_str = "_".join(pattern_arr)
	num_of_train_samples = options.num_of_train_samples
	num_of_test_samples = options.num_of_test_samples
	len_of_sample = options.len_of_sample

	# address for saving
	save_train_data_filename = "EXP_%d/train_data_EXP%d_%s_%dmer_%d.data" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_train_samples)
	save_train_label_filename = "EXP_%d/train_label_EXP%d_%s_%dmer_%d.label" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_train_samples)
	save_test_data_filename = "EXP_%d/test_data_EXP%d_%s_%dmer_%d.data" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_test_samples)
	save_test_label_filename = "EXP_%d/test_label_EXP%d_%s_%dmer_%d.label" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_test_samples)

	##########################
	# Print settings
	##########################
	print ("Number of train samples: %d" %(num_of_train_samples))
	print ("Number of test samples: %d" %(num_of_test_samples))
	print ("Length of each sample: %d" %(len_of_sample))
	print ("Embeded patterns %s" %(pattern_arr_str))
	print ("Train data are saved in %s" %(save_train_data_filename))
	print ("Train label are saved in %s" %(save_train_label_filename))
	print ("Test data are saved in %s" %(save_test_data_filename))
	print ("Test label are saved in %s" %(save_test_label_filename))
	print ("=======================================================")

	##########################
	# Generate Data
	#########################
	tools.dataGenerator_kmer(save_train_data_filename, save_train_label_filename, num_of_train_samples, len_of_sample, pattern_arr, kmer)
	tools.dataGenerator_kmer(save_test_data_filename, save_test_label_filename, num_of_test_samples, len_of_sample, pattern_arr, kmer)

	print ("Finish the data generating.")


def generateData_v3():
	print ("Start to generate in silco data")
	print ("====START=======================================")
	##########################
	# read options and args
	###########################
	usage = "usage: %prog [options] <output_file_path>"
	parser = OptionParser(usage)

	# settings
	parser.add_option('--patterns', dest = 'patterns', default='GATAGATTTC,CAGCCAACTG', help='The patterns that are embeded in the positive cases. Separate using comma')
	parser.add_option('--numOfTrainSamples', dest='num_of_train_samples', default=10000, type='int', help='The number of samples used for training [Default: %default]')
	parser.add_option('--numOfTestSamples', dest='num_of_test_samples', default=10000, type='int', help='The number of samples used for testing [Default: %default]')
	parser.add_option('--lenOfSample', dest='len_of_sample', default=500, type='int', help='The length of samples [Default: %default]')
	parser.add_option('--probablityFunction', dest='prob_func', default="step_function", help='The function used to discribe the probability distribution (step_function, inverse) [Default: %default]')
	parser.add_option('--prob_threshold', dest='prob_threshold', default=30, type='int', help='Below this threshold, the Prob(+) = 1 [Default: %default]')
	

	(options, args) = parser.parse_args()

	##############################
	# Probability functions
	##############################
	def step_function(distance):
		'''
		Prob (+) = 1 when distance <= threshold
			 = 0 when distance >  threshold
		'''
		if distance <=  options.prob_threshold:
			return 1
		else:
			return 0

	def inverse(distance):
		'''
		Prob (+) = 1 when distance <= threshold
			 = 1/(distance-threshold) when distance > threshold
		'''
		if distance <= options.prob_threshold:
			return 1
		else:
			return 1/float(distance-options.prob_threshold)


	###########################
	# Default settings
	###########################
	exp_index = 3
	kmer = 4
	pattern_arr = options.patterns.split(",")
	pattern_arr_str = "_".join(pattern_arr)
	num_of_train_samples = options.num_of_train_samples
	num_of_test_samples = options.num_of_test_samples
	len_of_sample = options.len_of_sample
	prob_threshold = options.prob_threshold

	possibles = globals().copy()
	possibles.update(locals())
	prob_func = possibles.get(options.prob_func)

	# address for saving
	save_train_data_filename = "EXP_%d/train_data_EXP%d_%s_%dmer_%d.data" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_train_samples)
	save_train_label_filename = "EXP_%d/train_label_EXP%d_%s_%dmer_%d.label" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_train_samples)
	save_test_data_filename = "EXP_%d/test_data_EXP%d_%s_%dmer_%d.data" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_test_samples)
	save_test_label_filename = "EXP_%d/test_label_EXP%d_%s_%dmer_%d.label" %(exp_index, exp_index, pattern_arr_str, kmer, num_of_test_samples)

	##########################
	# Print settings
	##########################
	print ("Number of train samples: %d" %(num_of_train_samples))
	print ("Number of test samples: %d" %(num_of_test_samples))
	print ("Length of each sample: %d" %(len_of_sample))
	print ("Embeded patterns: %s" %(pattern_arr_str))
	print ("Probability function: %s" %(prob_func))
	print ("Prob(+) = 1 when below threshold: %s" %(prob_threshold))
	print ("Train data are saved in %s" %(save_train_data_filename))
	print ("Train label are saved in %s" %(save_train_label_filename))
	print ("Test data are saved in %s" %(save_test_data_filename))
	print ("Test label are saved in %s" %(save_test_label_filename))
	print ("=======================================================")

	##########################
	# Generate Data
	#########################
	tools.dataGenerator_sign_2feature(save_train_data_filename, save_train_label_filename, num_of_train_samples, len_of_sample, pattern_arr, kmer, prob_func, prob_threshold)
	tools.dataGenerator_sign_2feature(save_test_data_filename, save_test_label_filename, num_of_test_samples, len_of_sample, pattern_arr, kmer, prob_func, prob_threshold)

	print ("Finish the data generating.")

def generateData_v4():
	print ("Start to generate in silco data")
	print ("====START=======================================")
	##########################
	# read options and args
	###########################
	usage = "usage: %prog [options] <output_file_path>"
	parser = OptionParser(usage)

	# settings
	parser.add_option('--patterns', dest = 'patterns', default='GATAGATTTC,CAGCCAACTG', help='The patterns that are embeded in the positive cases. Separate using comma')
	parser.add_option('--numOfTrainSamples', dest='num_of_train_samples', default=10000, type='int', help='The number of samples used for training [Default: %default]')
	parser.add_option('--numOfTestSamples', dest='num_of_test_samples', default=10000, type='int', help='The number of samples used for testing [Default: %default]')
	parser.add_option('--lenOfSample', dest='len_of_sample', default=500, type='int', help='The length of samples [Default: %default]')
	parser.add_option('--probablityFunction', dest='prob_func', default="step_function", help='The function used to discribe the probability distribution (step_function, inverse) [Default: %default]')
	parser.add_option('--prob_threshold', dest='prob_threshold', default=30, type='int', help='Below this threshold, the Prob(+) = 1 [Default: %default]')
	

	(options, args) = parser.parse_args()

	##############################
	# Probability functions
	##############################
	def step_function(distance):
		'''
		Prob (+) = 1 when distance <= threshold
			 = 0 when distance >  threshold
		'''
		if distance <=  options.prob_threshold:
			return 1
		else:
			return 0

	def inverse(distance):
		'''
		Prob (+) = 1 when distance <= threshold
			 = 1/(distance-threshold) when distance > threshold
		'''
		if distance <= options.prob_threshold:
			return 1
		else:
			return 1/float(distance-options.prob_threshold)


	###########################
	# Default settings
	###########################
	exp_index = 4
	pattern_arr = options.patterns.split(",")
	pattern_arr_str = "_".join(pattern_arr)
	num_of_train_samples = options.num_of_train_samples
	num_of_test_samples = options.num_of_test_samples
	len_of_sample = options.len_of_sample
	prob_threshold = options.prob_threshold

	possibles = globals().copy()
	possibles.update(locals())
	prob_func = possibles.get(options.prob_func)

	# address for saving
	save_train_data_filename = "EXP_%d/train_data_EXP%d_%s_onehot_%d.data" %(exp_index, exp_index, pattern_arr_str, num_of_train_samples)
	save_train_label_filename = "EXP_%d/train_label_EXP%d_%s_onehot_%d.label" %(exp_index, exp_index, pattern_arr_str, num_of_train_samples)
	save_test_data_filename = "EXP_%d/test_data_EXP%d_%s_onehot_%d.data" %(exp_index, exp_index, pattern_arr_str, num_of_test_samples)
	save_test_label_filename = "EXP_%d/test_label_EXP%d_%s_onehot_%d.label" %(exp_index, exp_index, pattern_arr_str, num_of_test_samples)

	##########################
	# Print settings
	##########################
	print ("Number of train samples: %d" %(num_of_train_samples))
	print ("Number of test samples: %d" %(num_of_test_samples))
	print ("Length of each sample: %d" %(len_of_sample))
	print ("Embeded patterns: %s" %(pattern_arr_str))
	print ("Probability function: %s" %(prob_func))
	print ("Prob(+) = 1 when below threshold: %s" %(prob_threshold))
	print ("Train data are saved in %s" %(save_train_data_filename))
	print ("Train label are saved in %s" %(save_train_label_filename))
	print ("Test data are saved in %s" %(save_test_data_filename))
	print ("Test label are saved in %s" %(save_test_label_filename))
	print ("=======================================================")

	##########################
	# Generate Data
	#########################
	tools.dataGenerator_sign_2feature_onehot(save_train_data_filename, save_train_label_filename, num_of_train_samples, len_of_sample, pattern_arr, prob_func, prob_threshold)
	tools.dataGenerator_sign_2feature_onehot(save_test_data_filename, save_test_label_filename, num_of_test_samples, len_of_sample, pattern_arr, prob_func, prob_threshold)

	print ("Finish the data generating.")

###########################################################################
if __name__ == "__main__":
	#generateData_v1()
	#generateData_v2()
	#generateData_v3()
	generateData_v4()

###END#################
