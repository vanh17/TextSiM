import sys


''' This file is to test if we remove unnecessary information such as parsing will the model from Hugging Face still works.
	Update on May 1st, 2021: Yes, it still works, they only worry about the label and the sentence pairs not their parsing results. 
'''
'''usage: we need to pass:
	1. original_file (in the MNLI format)
	2. simplified_sent1: file of simplified first sentences in the pairs from original file
	3. simplified_sent2: file of simplified second sentences in the pairs from the original file
	4. new_file: new file name with full directory to save updated data.
'''
def main():
	original_file = open(sys.argv[1], "r").readlines()
	simplified_sent1 = open(sys.argv[2], "r").readlines()
	simplified_sent2 = open(sys.argv[3], "r").readlines()
	new_file = sys.argv[4]
	'''we need to pass the original file first, each files contains one-line samples and hence we need to remove a couple of things
		1. binary parse for two sentences
		2. parse for two sentences
		3. Replace original sentences with simplified sentences
	''' 
	with open(new_file, "w") as updated:
		# copy over the header, nothing change here
		updated.write(original_file[0])
		num_samples = len(original_file)
		for i in range(1, num_samples):
			original_sample = original_file[i].split("\t")
			sent1 = simplified_sent1[i-1].strip("\n")
			sent2 = simplified_sent2[i-1].strip("\n")
			# start removing unnecessary information in the original sample, leave them blank
			original_sample[4] = ""
			original_sample[5] = ""
			original_sample[6] = ""
			original_sample[7] = ""
			# replace simplified senteces with original sentences
			original_sample[8] = sent1
			original_sample[9] = sent2
			# write simplified sample to new file.
			updated.write("\t".join(original_sample))

if __name__ == '__main__':
	main()