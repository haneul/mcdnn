
mean_interval = 5. #5s
T = 60*5 #5min 

#proportion = 0.001
proportion = 0
total_n = 200

import random
# S1
target_n = 7
p = 0.65
in_duration = 12*T

# S2
target_n = 4
p = 0.9
in_duration = 24*T


until = 36000
current = 0
in_ = False
res = []
inout = [(0, False)]
while True:
    current += random.expovariate(1/mean_interval)
    if current > until: break
    if not in_:
        dice = random.random()
        if dice < proportion:
            in_ = True
            out_time = current + in_duration
            inout.append((current, in_))
    else:
        if current > out_time:
            in_ = False
            inout.append((current, in_))
    if not in_:
        pick = random.randint(0,199)
    else:
        dice = random.random()
        if dice < p:
            pick = random.randint(0, target_n-1)
        else:
            pick = random.randint(target_n, 199)
    res.append((current, pick))

import pickle
with open("special_trace.pcl", "wb") as f:
    pickle.dump(res, f)

import matplotlib.pyplot as plt 
from mpltools import style
style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(211)
ax.step(map(lambda x:x[0], inout), map(lambda x:x[1], inout), where='post')
plt.xlim(0,36000)
plt.ylim(-0.2,1.2)
plt.ylabel("Specialized")
plt.yticks([0,1], ['out','in'])
ax = fig.add_subplot(212)
ax.scatter(map(lambda x:x[0], res), map(lambda x:x[1], res))
plt.ylabel("person ID")
plt.xlabel("time")
plt.ylim(0,200)
plt.xlim(0,36000)
#plt.show()
fig.savefig('specialized.pdf', bbox_inches='tight')
