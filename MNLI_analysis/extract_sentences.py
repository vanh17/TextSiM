import sys

def main():
	original_file = sys.argv[1]
	sentence1_only_file = sys.argv[2]
	sentence2_only_file = sys.argv[3]
	with open(original_file, "r") as original:
		with open(sentence1_only_file, "w") as output1:
			with open(sentence2_only_file, "w") as output2:
				for line in original:
					line = line.split("\t")
					output1.write(line[8]+"\n")
					output2.write(line[9]+"\n")

if __name__ == '__main__':
	main()