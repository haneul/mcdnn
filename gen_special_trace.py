
mean_interval = 5. #5s
T = 60*5 #5min 

proportion = 0.001 
total_n = 200

import random
target_n = 14
p = 0.8

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
            out_time = current + 5*T
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
            pick = random.randint(0, target_n)
        else:
            pick = random.randint(target_n+1, 199)
    res.append((current, pick))


import matplotlib.pyplot as plt 
from mpltools import style
style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(211)
ax.step(map(lambda x:x[0], inout), map(lambda x:x[1], inout), where='post')
plt.xlim(0,10800)
plt.ylim(-0.2,1.2)
plt.ylabel("Specialized")
plt.yticks([0,1], ['out','in'])
ax = fig.add_subplot(212)
ax.scatter(map(lambda x:x[0], res), map(lambda x:x[1], res))
plt.ylabel("person ID")
plt.xlabel("time")
plt.ylim(0,200)
plt.xlim(0,10800)
plt.show()
fig.savefig('specialized.pdf', bbox_inches='tight')

