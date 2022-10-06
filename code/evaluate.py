import json
answer_list = json.load(open("./khistory_25_cos.json", "r"))
mm = -1
for d in answer_list:
	d=d["answer"]
	correct = 0
	all = 0
	for k in d.keys():
		max_num = -1
		max_class = None
		for kk in d[k].keys():
			all += d[k][kk]
			if d[k][kk] > max_num:
				max_class = kk
				max_num = d[k][kk]
		correct += max_num
	mm = max(mm, correct/all)
	# print(correct/all)
print(mm)