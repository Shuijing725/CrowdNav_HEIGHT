import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# change the displayed name of plotted models here
legends = ['ours, rand env', '', '']

# add more training curves by directory name here!
log_list = [
pd.read_csv('trained_models/ours_HH_RH_randEnv/progress.csv'),
# pd.read_csv('data/ours_RH_HH_loungeEnv_resumeFromRand/progress.csv'),
		   ]

logDicts = {}
for i in range(len(log_list)):
	logDicts[i] = log_list[i]

graphDicts={0:'eprewmean', 1:'loss/value_loss'}

legendList=[]
# summarize history for accuracy

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0
	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.plot(logDicts[key]['misc/total_timesteps'], logDicts[key][graphDicts[i]])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='lower right')
	legendList=[]



plt.show()


