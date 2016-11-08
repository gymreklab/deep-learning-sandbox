'''
This file provides auxiliary tools for loading data

main function:
(1)load_kmer()
(2)load_onehot()
'''
import numpy

PATH = "/home/anz023/noncoding_predict/Noncoding_Feature_Distance_Detector/data_in_silco/"

def create_dictionary(data1, data2):
	word_dict = {}
	word_arr = []
	count = 0

	# data1
	ifile = open(data1)
	for line in ifile:
		elems = line.split()
		for elem in elems:
			if elem not in word_dict:
				word_dict[elem] = count
				word_arr.append(elem)
				count += 1
	ifile.close()
	
	# data2
	ifile = open(data2)
	for line in ifile:
		elems = line.split()
		for elem in elems:
			if elem not in word_dict:
				word_dict[elem] = count
				word_arr.append(elem)
				count += 1
	ifile.close()
	return word_dict, word_arr, count

def read_from_file(filepath, key_arr, dataOrLabel):
	data = []
	ifile = open(filepath)
	for line in ifile:
		if dataOrLabel == "data":
			elems = line.split()
			elems_IDX = [key_arr.index(elem) for elem in elems]
			data.append(elems_IDX)
			#elems_np = numpy.array(elems_IDX)
			#data.append(elems_np)
		else:
			line = line.strip()
			elems_IDX = key_arr.index(line)
			data.append(elems_IDX)
	ifile.close()
	return data

def read_from_file_3D(filepath):
	data = []
	ifile = open(filepath)
	for line in ifile:
		# elems: [0010 0100 1000 0010 ...]
		# elem: 0010
		elems = line.split()
		elems_matrix = [[int(digit) for digit in elem] for elem in elems]
		data.append(elems_matrix)

	ifile.close()
	return data

def load_data(data_path, label_path, NT):
	data = read_from_file(data_path, NT, "data")
	LABEL = ['0', '1']
	label = read_from_file(label_path, LABEL, "label")
	return (data, label)

def load_3D_data(data_path, label_path):
	data = read_from_file_3D(data_path)
	LABEL = ['0', '1']
	label = read_from_file(label_path, LABEL, "label")
	return (data, label)

def load_kmer():
	train_data = "%s/EXP_3/train_data_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.data" %PATH
	train_label = "%s/EXP_3/train_label_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.label" %PATH
	test_data = "%s/EXP_3/test_data_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.data" %PATH
	test_label = "%s/EXP_3/test_label_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.label" %PATH

	# create dictionary
	w2idx, NT, max_feature = create_dictionary(train_data, test_data)

	# load data and label
	train_set = load_data(train_data, train_label, NT)
	test_set = load_data(test_data, test_label, NT)


	return train_set, test_set, max_feature

def load_onehot():
	train_data = "%s/EXP_4/train_data_EXP4_GATAGATTTC_CAGCCAACTG_onehot_10000.data" %PATH
	train_label = "%s/EXP_4/train_label_EXP4_GATAGATTTC_CAGCCAACTG_onehot_10000.label" %PATH
	test_data = "%s/EXP_4/test_data_EXP4_GATAGATTTC_CAGCCAACTG_onehot_10000.data" %PATH
	test_label = "%s/EXP_4/test_label_EXP4_GATAGATTTC_CAGCCAACTG_onehot_10000.label" %PATH

	train_set = load_3D_data(train_data, train_label)
	test_set = load_3D_data(test_data, test_label)

	return train_set, test_set
###END#######################################
