from tqdm import tqdm
import json
import torch
from collections import defaultdict
import random
import sys
random.seed(2123142)
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from torchvision import datasets
from tqdm import tqdm
import json
import torch
from collections import defaultdict
import random
import sys
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5,rc={"lines.linewidth":2.5})
def get_class(a, b):
	ans = torch.cdist(a.unsqueeze(0).float(), b.float(), p=2)
	# ans = torch.cosine_similarity(a.unsqueeze(0).float(),b.float(),dim=1)
	# return ans.max(dim=0)[1].item()
	return ans.min(dim=1)[1].item()
K = 10
palette = np.array(sns.color_palette("hls",K))
train_data = datasets.MNIST(root="./data/",train=True,download=True)
data = train_data.data
label = train_data.targets
number = data.shape[0]
data = data.reshape(number,-1).float()
label = label.float()
index_vec = {}
l = []
while True:
	i = random.randint(0, 60000-1)
	if i not in l:
		l.append(i)
	if len(l) == K:
		break
center = torch.index_select(data,0,torch.tensor(l))
last_center = None
ts = manifold.TSNE(n_components=2,perplexity=50.0,n_iter=2000)
history = []
while True:
	sum = 0
	label_list = []
	last_center = center
	index_vec = defaultdict(list)
	index_label = defaultdict(list)
	for i in tqdm(range(number)):
		c = get_class(data[i],center)
		label_list.append(c)
		index_vec[c].append(data[i].unsqueeze(0))
		index_label[c].append(label[i].item())
	center = torch.tensor([])
	for i in range(K):
		center = torch.cat((center, torch.mean(torch.cat(index_vec[i],dim=0),dim=0).unsqueeze(0)),dim=0)
	for i in range(center.shape[0]):
		sum += torch.cdist(last_center[i].unsqueeze(0), center[i].unsqueeze(0), p=2).item()
		# print(torch.cdist(last_center[i].unsqueeze(0), center[i].unsqueeze(0), p=2).item(), end = " ")
	print(sum)
	if sum < 1:
		print("over")
		y=ts.fit_transform(data)
		plt.scatter(y[:,0],y[:,1],c=palette[label_list])
		plt.savefig("vis_50_2000.png")
		print(label_list)
		json.dump({"label_list":label_list},open("./label_list.json","w"))
		break
	answer = {}
	for k in index_label.keys():
		d = defaultdict(int)
		for i in index_label[k]:
			d[int(i)]+=1
		answer[k] = d
	history.append({"answer":answer})
	json.dump(history, open("./khistory_10.json","w"))