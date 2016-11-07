import numpy

PATH = "/home/anz023/noncoding_predict/Noncoding_Feature_Distance_Detector/data_in_silco/"
#NT = ['A','C','T','G']
#LABEL = ['0', '1']


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

def load_data(data_path, label_path, NT):
	data = read_from_file(data_path, NT, "data")
	LABEL = ['0', '1']
	label = read_from_file(label_path, LABEL, "label")
	return (data, label)

def atisfull():
	train_data = "%s/EXP_3/train_data_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.data" %PATH
	train_label = "%s/EXP_3/train_label_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.label" %PATH
	test_data = "%s/EXP_3/test_data_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.data" %PATH
	test_label = "%s/EXP_3/test_label_EXP3_GATAGATTTC_CAGCCAACTG_4mer_10000.label" %PATH

	# create dictionary
	# w2idx = {'A':0, 'C':1, 'T':2, 'G':3}
	# labels2idx = {'0':0 , '1':1}
	w2idx, NT, max_feature = create_dictionary(train_data, test_data)
	#labels2idx, LABEL = create_dictionary(train_label, test_label)
	#dicts = {'words2idx': w2idx, 'labels2idx':labels2idx}

	# load data and label
	train_set = load_data(train_data, train_label, NT)
	test_set = load_data(test_data, test_label, NT)


	return train_set, test_set, max_feature
