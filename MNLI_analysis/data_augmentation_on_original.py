import sys
from random import random



'''
	This is the main file where we create simplified train/dev sets for MNLI experiments. 
	We use random to get add simplified data to the original or totally replace original data with x% of simplified data
	We reported mean value and standard deviation for N = 3 runs for these experiements.
'''

def remove_parses(original_sample):
	'''
		This function removes unnecessary information such binanry parses and general parses come with the MNLI dataset
		Do nothing much, just set the current value to empty string and then return the original_sample
	'''
	original_sample[4] = ""
	original_sample[5] = ""
	original_sample[6] = ""
	original_sample[7] = ""
	return original_sample	

def add_sample(original_sample, sent1, sent2):
	'''
		This function add simplified original that corresponding with the original sample to it
		Return the 2 strings, one is the original sample, and the other is its simplified counterpart
		Take three parameters: 
		1. original_sample: in the format of list, each element is one of the information in original sample
		2. sent1: the simplified version of the sentence 1
		3. sent2: the simplified version of the sentence 2
	'''
	# make the copy of the original sample so that they will link together. They now will two separate file
	simplified = original_sample.copy()
	# change the sentences to their simplified counterparts.
	simplified[8] = sent1
	simplified[9] = sent2
	return ["\t".join(original_sample), "\t".join(simplified)]


def replace_sample(original_sample, sent1, sent2):
	'''
		This function replace the original with simplified sample.
		Take three parameters: 
		1. original_sample: in the format of list, each element is one of the information in original sample
		2. sent1: the simplified version of the sentence 1
		3. sent2: the simplified version of the sentence 2
	'''
	original_sample[8] = sent1
	original_sample[9] = sent2
	return ["\t".join(original_sample)]


'''usage: we need to pass:
	1. original_file (in the MNLI format)
	2. simplified_sent1: file of simplified first sentences in the pairs from original file
	3. simplified_sent2: file of simplified second sentences in the pairs from the original file
	4. new_file: new file name with full directory to save updated data.
	5. add_or_replace: to flag if we want to add simplified data or simply replace the original data
	6. threshold: x% of simplified data used in this altered train/set data sets.
'''
def main():
	original_file = open(sys.argv[1], "r").readlines()
	simplified_sent1 = open(sys.argv[2], "r").readlines()
	simplified_sent2 = open(sys.argv[3], "r").readlines()
	new_file = sys.argv[4]
	add_or_replace = int(sys.argv[5]) # 0 mean add simplified samples to the original dataset, 1 mean to replace it with the original
	threshold = float(sys.argv[6])
	'''we need to pass the original file first, each files contains one-line samples and hence we need to remove a couple of things
		1. binary parse for two sentences
		2. parse for two sentences
		3. Replace original sentences with simplified sentences
		The above only happen if random > a threshold.
	''' 
	count = 0
	with open(new_file, "w") as updated:
		# copy over the header, nothing change here
		updated.write(original_file[0])
		num_samples = len(original_file)
		for i in range(1, num_samples):
			original_sample = original_file[i].split("\t")
			sent1 = simplified_sent1[i-1].strip("\n")
			sent2 = simplified_sent2[i-1].strip("\n")
			# start removing unnecessary information in the original sample, leave them blank
			original_sample = remove_parses(original_sample)
			samples = ["\t".join(original_sample)]
			if add_or_replace == 0:
				if random() <= threshold:
					samples = add_sample(original_sample, sent1, sent2)
			else:
				if random() <= threshold:
					samples = replace_sample(original_sample, sent1, sent2) 
			# write simplified sample to new file.
			for sample in samples:
				if "\n" not in sample:
					sample += "\n"
				updated.write(sample)
				count += 1
		print(count)

if __name__ == '__main__':
	main()