import json
import sys
import random

def main():
	# usage: python3 extract_sentences.py tacred_file.json sentences.txt info.txt
	# opening the json file
	file = open(sys.argv[1], "r")

	# return json object (list of dictionary)
	data = json.load(file)

	# comment this out when to extract the whole sentences.
	# This is for the quick analysis of what the statistic are.
	data = random.sample(data, 100)

	#initial printing to see what the structure of tacred_file for extraction
	print(len(data))
	# output: number of instances in the file. 
	# each entry is a dictionary of keys
	print(data[0].keys())
	# output :'id', 'docid', 'relation', 'token', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 
	#         'subj_type', 'obj_type', 'stanford_pos', 'stanford_ner', 'stanford_head', 'stanford_deprel'
	# ENDED printing important information.
	
	# this is to traverse through the list of instances but mostly focuses on the token.
	# 1. The main goal is to check if the length of the simplified sentences are shorter than the original sentences
	# 2. Also if the neural simplification approach reserver important information (Subject and Object of the relation)
	# 3. We also want to know the percentage of the text in here was simplified by the neural approach. And among them
	# 4. How many of them do reserve the important information
	# Therefore, this will create 2 files: 
	#        1. sentences.txt (to run through neural TS) by concatenating all tokens with space. Seperated by "\n"
	#        2. info.txt in the format of subject (strings) \t object (strings) \t length (integer). 
	#           Each sentence is on seperate line
	# create the list to hold sentences and info.
	sentences = [] # list of strings
	info = [] # list of tuples formatted as {subject, object, length}
	for sample in data:
		sentences.append(" ".join(sample["token"])) # join all the tokens in the list named token for each sample
		subj = " ".join(sample["token"][sample["subj_start"]:sample["subj_end"]+1])
		obj = " ".join(sample["token"][sample["obj_start"]:sample["obj_end"]+1])
		info.append((subj, obj, len(sample["token"])))
	# check if we match all the sentences with the sample numbers.
	assert len(sentences) == len(info)
	assert len(sentences) == len(data)
	
	# close the file
	file.close()

	# write the info to the output files
	with open(sys.argv[2], 'w') as sentencefile:
		for sent in sentences:
			sentencefile.write(sent + "\n")

	# write the info to the output files
	with open(sys.argv[3], 'w') as infofile:
		for tup in info:
			infofile.write(tup[0] + "\t" + tup[1] + "\t" + str(tup[2]) + "\n")


# call main for function to start:
if __name__ == '__main__':
	main()