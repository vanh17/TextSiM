import json
import sys
import random

def print_out_analysis(analysis):
	total = len(analysis)
	subj = len([i for i in analysis if i[0] > 0])
	obj = len([i for i in analysis if i[1] > 0])
	trig = len([i for i in analysis if i[2] > 0])
	longer = len([i for i in analysis if i[3] > 0])
	shorter = len([i for i in analysis if i[3] < 0])
	same = len([i for i in analysis if i[3] == 0 and i[4] == 1])
	no_sim = len([i for i in analysis if i[3] == 0 and i[4]  == 0])
	print("Number of subj reserved:", subj/total)
	print("Number of obj reserved:", obj/total)
	print("Number of triggers reserved:", trig/total)
	print("Number of longer simplification:", longer/total)
	print("Number of shorter simplification:", shorter/total)
	print("Number of same length simplification:", same/total)
	print("Number of no simplified:", no_sim/total)
	subj = len([i for i in analysis if i[0] > 0 and i[3] > 0])
	print("Number of subj reserved with longer simplification:", subj/longer)
	obj = len([i for i in analysis if i[1] > 0 and i[3] > 0])
	print("Number of obj reserved with longer simplification:", obj/longer)
	trig = len([i for i in analysis if i[2] > 0 and i[3] > 0])
	print("Number of triggers reserved with longer simplification:", trig/longer)
	subj = len([i for i in analysis if i[0] > 0 and i[3] < 0])
	print("Number of subj reserved with shorter simplification:", subj/shorter)
	obj = len([i for i in analysis if i[1] > 0 and i[3] < 0])
	print("Number of obj reserved with shorter simplification:", obj/shorter)
	trig = len([i for i in analysis if i[2] > 0 and i[3] < 0])
	print("Number of triggers reserved with shorter simplification:", trig/shorter)
	subj = len([i for i in analysis if i[0] > 0 and i[3] == 0 and i[4] > 0])
	print("Number of subj reserved with same length simplification:", subj/same)
	obj = len([i for i in analysis if i[1] > 0 and i[3] == 0 and i[4] > 0])
	print("Number of obj reserved with same length simplification:", obj/same)
	trig = len([i for i in analysis if i[2] > 0 and i[3] == 0 and i[4] > 0])
	print("Number of triggers reserved with same length simplification:", trig/same)

def main():
	# usage: python3 tacred_statistics.py tacred_sim_sents.json tacred_info.txt trigger.txt analysis.txt original_sents.txt
	# opening the json file for simplified sentences, subj_obj information, and triggers for relations
	sim_sents = [i.strip("\n") for i in open(sys.argv[1], "r").readlines()]
	org_sents = [i.strip("\n") for i in open(sys.argv[4], "r").readlines()]
	info = [i.strip("\n").split("\t") for i in open(sys.argv[2], "r").readlines()]
	triggers = [i.strip("\n") for i in open(sys.argv[3], "r").readlines()]


	# check if the numbers of simplified senteces is the same as info and triggers. Remember that, one trigger 
	# for each original sentence.
	assert len(sim_sents) == len(info)
	assert len(sim_sents) == len(triggers)

	# we will store 5 information for each instance in the format of the tuple of 4: 
	#	1. If subject is reserved, 1 else, 0
	# 	2. If object is reserved, 1 else, 0
	# 	3. If Trigger is reserved, 1 else, 0
	# 	4. If the length of simplified sentence is longer than the original, 1, same, 0, shorter, -1
	# 	5. If the sentence is simplified or not, 1 and 0
	analysis = []
	
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
	for i in range(len(sim_sents)):
		sent = sim_sents[i].split()
		# if the simplification ever existed
		isSim = 0
		if sim_sents[i] != org_sents[i]:
			isSim = 1
		# set the len differences
		l = len(sent) - int(info[i][2])
		subj = 0
		if info[i][0] in sim_sents[i]:
			subj = 1
		obj = 0
		if info[i][1] in sim_sents[i]:
			obj = 1
		trig = 0
		if triggers[i] in sim_sents[i]:
			trig = 1
		analysis.append((subj, obj, trig, l, isSim))
	# check if we match all the sentences with the sample numbers.
	assert len(analysis) == len(info)
	print_out_analysis(analysis)

	# write the info to the output files
	with open(sys.argv[4], 'w') as output:
		for sent in analysis:
			output.write(str(sent[0]) + "\t" + str(sent[1]) + "\t" + str(sent[2]) + "\t" + str(sent[3]) +"\n")

# call main for function to start:
if __name__ == '__main__':
	main()