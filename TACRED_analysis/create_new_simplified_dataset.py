# create_new_simplified_dataset.py
# This file will take the extracted sentences after simplification and 
# make a new dataset for the original training/evaluating codes from other papers

# import neccessary libraries
import sys
import json

#helper to find all occurences of substring
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

def get_start_end(sentence, subj, obj): #(sentence, word):
	# Old version find start, end of subject and object seperately.
	# word = word.split()
	# sentence = sentence.split()
	# start = sentence.index(word[0])
	# end = sentence.index(word[0])
	# itr = 1
	# prev = start
	# while itr < len(word):
	# 	end = sentence.index(word[itr], prev+1) 
	# 	if end == prev + 1:
	# 		itr += 1
	# 		prev = end
	# 	else:
	# 		start = sentence.index(word[0], start+1)
	# 		prev= start
	# 		itr = 1
	# New version, give the obj and subj, we will try to return the position of obj and
	# subj that has a smallest distance between the two. 
	space_pos = list(find_all(sentence, " ")) # to find all the positions of whitespaces we have
	# this to determine the relative positon of the subj and obj. 
	start_subj = list(find_all(sentence, subj))
	start_obj = list(find_all(sentence, obj))
	subj_pos = start_subj[0]
	obj_pos = start_obj[0]
	shortest_distance = abs(start_subj[0] - start_obj[0])
	for subj_s in start_subj:
		for obj_s in start_obj:
			if abs(subj_s - obj_s) < shortest_distance:
				shortest_distance = abs(subj_s - obj_s)
				subj_pos = subj_s
				obj_pos = obj_s
	# now we have the start of subj and obj that satisfy the shortest distance between them
	# We will use this absolute index (character-based) to get the relative index (word-based)
	# We do this by compare the absolute start with the each whitespace and accumulate the relative
	# index until we make it to where the whitespace is one before the start index.
	rel_subj_start = 0
	rel_obj_start = 0
	for pos in space_pos:
		if subj_pos > pos: # keep add one if the subj start position is behind the current whitespace position
			rel_subj_start += 1
		if obj_pos > pos: # keep add one if the obj start position is behind the current whitespace position
			rel_obj_start += 1
	return rel_subj_start, (rel_subj_start + len(subj.split())-1), rel_obj_start, (rel_obj_start + len(obj.split())-1) 

def create_sample(sentence, tuple_subj_obj, original, pos_rels):
	# do something here
	# keys: id, docid, relation, token, subj_start, subj_end, obj_start, obj_end, subj_type, obj_type
	# copy over the unchanged information from the original sentence
	sample = {"id": original["id"], "relation": original["relation"], "docid" : original["docid"], "subj_type" : original["subj_type"], "obj_type" : original["obj_type"]}
	# create new list of tokens for new sample
	sample["token"] = sentence.split()
	# get the start, end for subj, obj
	sample["subj_start"], sample["subj_end"], sample["obj_start"], sample["obj_end"] = get_start_end(sentence, tuple_subj_obj[0], tuple_subj_obj[1])
	# get stanford_pos:
	sample["stanford_pos"] = pos_rels[0].split(" ")
	# get stanford_ner
	# convert to old stanford_ner
	old_ner_with_location = [ner if ner not in ["CITY", "STATE_OR_PROVINCE", "COUNTRY"] else "LOCATION" for ner in pos_rels[1].split(" ")]
	old_ner = [ner if ner in ["PERSON", "LOCATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME", "DURATION", "SET"] else "O" for ner in old_ner_with_location]
	sample["stanford_ner"] = old_ner
	# get stanford_deprel
	sample["stanford_deprel"] = pos_rels[2].split(" ")
	# get stanford_head
	sample["stanford_head"] = pos_rels[3].split(" ")
	# check if the stanford linguistic information lengths match with the number of tokens. If not, we will not have
	# a valid dataset to run TACRED official code. Check if they are valid samples
	isValid = len(sample["stanford_deprel"]) == len(sample["stanford_head"]) and len(sample["stanford_deprel"]) == len(sample["stanford_ner"]) and len(sample["stanford_ner"]) == len(sample["stanford_pos"]) and len(sample["token"]) == len(sample["stanford_pos"])
	# if passes the asserts, simple return sample for simplified dataset.
	return (sample, isValid)

def main():
	# usage: python3 create_new_simplified_dataset.py [tacred_sim_sentences.txt] [tacred_info.txt] 
	#                                      [original.json] [tacred_sim_pos_rels.txt][tacred_simplified.json]
	# 	1. The first argument is the file storing simplified sentences from designated Text Simplifcation system
	# 	2. The second argument is the file storing information about subj, obj associated with the above sentence
	# 	3. The third argument is the original file.
	#	4. The output of dataset in the format of TACRED.
	# open all the input files
	sentences = [line.strip("\n") for line in open(sys.argv[1], "r").readlines()]
	info = [line.strip("\n").split("\t") for line in open(sys.argv[2], "r").readlines()]
	original = json.load(open(sys.argv[3], "r"))
	# This will be a list of array of 3: [[POS, NER, DEPREL]]
	pos_rels = [line.strip("\n").split("\t") for line in open(sys.argv[4], "r").readlines()]

	# check if the original files and simplified sentences have the same length
	assert len(sentences) == len(original)
	assert len(original) == len(info)
	assert len(original) == len(pos_rels)

	# Initialize the cases so that we know if we want to add simplified data and its corresponding original
	addSim = 1
	addCorespOri = 0
	# Initialize the case here so that it know if we want to add Original sentences that does not have valid
	# simplification (does not preserve th subj and obj)
	addCompOri = 1
	# initialize holder 
	simplified = [] 
	# traverse each line of the input files to create samples of the new simplified dataset
	for i in range(len(sentences)):
		# check if the simplified sentences still consist important information for TACRED task
		if " "+info[i][0]+" " in sentences[i] and " "+info[i][1]+" " in sentences[i]:
			# if so, then it is valid and be added to the simplified dataset.
			# Also, we want to add the validated samples and ignore wrong format one
			if addSim:
				sample, isValid = create_sample(sentences[i], info[i], original[i], pos_rels[i])
				if isValid: 
					simplified.append(sample)
			if addCorespOri:
				simplified.append(original[i])
		# add the rest of the original samples that do no have valid simplification counterpart
		else:
			if addCompOri:
				simplified.append(original[i])

	print("Number of simplified samples:", len(simplified))

	# test code on original data create back to original data
	print(len(simplified))

	# now write new dataset set to the output file
	with open(sys.argv[5], "w") as output:
		json.dump(simplified, output)

if __name__ == '__main__':
	main()

