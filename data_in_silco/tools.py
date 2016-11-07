'''
This file contains some calculation tools for in silco data generation
'''
import random
import sys
import numpy
def dataGenerator(save_data_file, save_label_file, num_of_samples, len_of_sample, pattern_arr):
	'''
	generate and save the in silco data with random surrounding contexts
	'''
	#######################
	# randomized init
	#######################
	data = [['A' for _1 in range(len_of_sample)] for _2 in range(num_of_samples)]
	label = [[0 for _1 in range(len_of_sample)]  for _2 in range(num_of_samples)]

	for sample_ID in range(num_of_samples):
		for nt_ID in range(len_of_sample):
			nt = random.choice(['A','T','C','G'])
			data[sample_ID][nt_ID] = nt

	######################
	# embed the patterns
	######################
	for sample_ID in range(num_of_samples):
		total_pattern_length = sum([len(pttn) for pttn in pattern_arr])
		startPos = 0
		for pattern in pattern_arr:
			pattern_pos = random.choice([pos for pos in range(startPos, len_of_sample-total_pattern_length+1)])
			for nt_ID in range(len(pattern)):
				data[sample_ID][pattern_pos+nt_ID] = pattern[nt_ID]
				label[sample_ID][pattern_pos+nt_ID] = 1
			# update
			startPos = pattern_pos + len(pattern)
			total_pattern_length -= len(pattern)

	######################
	# save data
	######################
	save_arr(save_data_file, data, num_of_samples, len_of_sample)
	save_arr(save_label_file, label, num_of_samples, len_of_sample)


def dataGenerator_kmer(save_data_file, save_label_file, num_of_samples, len_of_sample, pattern_arr, kmer):
	'''
	generate and save the in silco data with random surrounding contexts
	'''
	#######################
	# randomized init
	#######################
	data = [['A' for _1 in range(len_of_sample)] for _2 in range(num_of_samples)]
	label = [['0' for _1 in range(len_of_sample)]  for _2 in range(num_of_samples)]

	for sample_ID in range(num_of_samples):
		for nt_ID in range(len_of_sample):
			nt = random.choice(['A','T','C','G'])
			data[sample_ID][nt_ID] = nt

	######################
	# embed the patterns
	######################
	for sample_ID in range(num_of_samples):
		total_pattern_length = sum([len(pttn) for pttn in pattern_arr])
		startPos = 0
		for pattern in pattern_arr:
			pattern_pos = random.choice([pos for pos in range(startPos, len_of_sample-total_pattern_length+1)])
			for nt_ID in range(len(pattern)):
				data[sample_ID][pattern_pos+nt_ID] = pattern[nt_ID]
				label[sample_ID][pattern_pos+nt_ID] = '1'
			# update
			startPos = pattern_pos + len(pattern)
			total_pattern_length -= len(pattern)

	#####################
	# make k-mers
	#####################
	data_kmer = [['NULL' for _1 in range(len_of_sample-kmer+1)] for _2 in range(num_of_samples)]
	label_kmer = [['NULL' for _1 in range(len_of_sample-kmer+1)] for _2 in range(num_of_samples)]
	for sample_ID in range(num_of_samples):
		for startPos in range(len_of_sample-kmer+1):
			data_str = ""
			label_str = ""
			for addPos in range(kmer):
				data_str += data[sample_ID][startPos+addPos]
				label_str += label[sample_ID][startPos+addPos]
			data_kmer[sample_ID][startPos] = data_str
			label_kmer[sample_ID][startPos] = label_str

	######################
	# save data
	######################
	save_arr(save_data_file, data_kmer, num_of_samples, len_of_sample-kmer+1)
	save_arr(save_label_file, label_kmer, num_of_samples, len_of_sample-kmer+1)

def dataGenerator_sign_2feature(save_data_file, save_label_file, num_of_samples, len_of_sample, pattern_arr, kmer, prob_function, prob_threshold):
	'''
	generate and save the in silco data with random surrounding contexts
	the labels are assigned randomly with the Prob
	'''
	#######################
	# randomized init
	#######################
	data = [['A' for _1 in range(len_of_sample)] for _2 in range(num_of_samples)]
	label = [['0']  for _2 in range(num_of_samples)]

	for sample_ID in range(num_of_samples):
		for nt_ID in range(len_of_sample):
			nt = random.choice(['A','T','C','G'])
			data[sample_ID][nt_ID] = nt

	######################
	# embed the patterns
	######################
	for sample_ID in range(num_of_samples):
		total_pattern_length = sum([len(pttn) for pttn in pattern_arr])
		in_or_out_threshold = random.choice(["within","outside"])
		if in_or_out_threshold == "within":
			feature_distance = random.randint(1,prob_threshold+1)
		else:
			feature_distance = random.randint((10*prob_threshold)+1, len_of_sample-total_pattern_length)
		prob_of_positive = prob_function(feature_distance)
		pos_or_neg = numpy.random.choice([0,1], p=[1-prob_of_positive, prob_of_positive])
		
		startPos = random.choice([pos for pos in range(0,len_of_sample-total_pattern_length-feature_distance+1)])
		# feature1
		pattern1 = pattern_arr[0]
		for nt_ID in range(len(pattern1)):
			data[sample_ID][startPos+nt_ID] = pattern1[nt_ID]
		# feature2
		pattern2 = pattern_arr[1]
		for nt_ID in range(len(pattern2)):
			data[sample_ID][startPos+len(pattern1)+feature_distance] = pattern2[nt_ID]

		# label
		label[sample_ID][0] = pos_or_neg

		print feature_distance, pos_or_neg
	#####################
	# make k-mers
	#####################
	data_kmer = [['NULL' for _1 in range(len_of_sample-kmer+1)] for _2 in range(num_of_samples)]
	for sample_ID in range(num_of_samples):
		for startPos in range(len_of_sample-kmer+1):
			data_str = ""
			for addPos in range(kmer):
				data_str += data[sample_ID][startPos+addPos]
			data_kmer[sample_ID][startPos] = data_str

	######################
	# save data
	######################
	save_arr(save_data_file, data_kmer, num_of_samples, len_of_sample-kmer+1)
	save_arr(save_label_file, label, num_of_samples, 1)


def dataGenerator_sign_2feature_onehot(save_data_file, save_label_file, num_of_samples, len_of_sample, pattern_arr, prob_function, prob_threshold):
	'''
	generate and save the in silco data with random surrounding contexts
	the labels are assigned randomly with the Prob
	'''
	#######################
	# randomized init
	#######################
	data = [['A' for _1 in range(len_of_sample)] for _2 in range(num_of_samples)]
	label = [['0']  for _2 in range(num_of_samples)]

	for sample_ID in range(num_of_samples):
		for nt_ID in range(len_of_sample):
			nt = random.choice(['A','T','C','G'])
			data[sample_ID][nt_ID] = nt

	######################
	# embed the patterns
	######################
	for sample_ID in range(num_of_samples):
		total_pattern_length = sum([len(pttn) for pttn in pattern_arr])
		in_or_out_threshold = random.choice(["within","outside"])
		if in_or_out_threshold == "within":
			feature_distance = random.randint(1,prob_threshold+1)
		else:
			feature_distance = random.randint((10*prob_threshold)+1, len_of_sample-total_pattern_length)
		prob_of_positive = prob_function(feature_distance)
		pos_or_neg = numpy.random.choice([0,1], p=[1-prob_of_positive, prob_of_positive])
		
		startPos = random.choice([pos for pos in range(0,len_of_sample-total_pattern_length-feature_distance+1)])
		# feature1
		pattern1 = pattern_arr[0]
		for nt_ID in range(len(pattern1)):
			data[sample_ID][startPos+nt_ID] = pattern1[nt_ID]
		# feature2
		pattern2 = pattern_arr[1]
		for nt_ID in range(len(pattern2)):
			data[sample_ID][startPos+len(pattern1)+feature_distance] = pattern2[nt_ID]

		# label
		label[sample_ID][0] = pos_or_neg

	#####################
	# make One-hot features
	#####################
	data_one_hot = [['NULL' for _1 in range(len_of_sample)] for _2 in range(num_of_samples)]
	for sample_ID in range(num_of_samples):
		for dataPos in range(len_of_sample):
			data_point_arr = [0, 0, 0, 0]
			nt_arr = ['A','T','C','G']
			data_point_arr[nt_arr.index(data[sample_ID][dataPos])] = 1
			data_one_hot[sample_ID][dataPos] = data_point_arr

	######################
	# save data
	######################
	# save_arr(save_data_file, data_kmer, num_of_samples, len_of_sample)
	# save_arr(save_label_file, label, num_of_samples, 1)


	#####################
	# return data
	####################
	return data_one_hot, label

def save_arr(filename, data, num_of_data, data_length):
	'''
	save a matrix in "data" into "filename"
	'''
	ofile = open(filename, 'w')
	for data_index in range(num_of_data):
		data_str = ""
		for feature_index in range(data_length):
			data_str += str(data[data_index][feature_index])+" "
		ofile.write(data_str+"\n")
	ofile.close()

###END########################################################
