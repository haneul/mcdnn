import model_pb2
import google.protobuf.text_format

param = model_pb2.ApplicationModel()
with open("model_sample.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)

#for model in param.models:
#    print(model)

energy_budget = 5.*3600 # 5Wh
#energy_budget = 2.*3600 # 5Wh
cost_budget = 0.0667 # dollar ($2/month)
#cost_budget = 0.04 # dollar ($2/month)
latency_limit = 1000 # 2secs
RTT = 100 # 100ms
server_cost = 1.8e-7 # $ per ms
send_energy = 3.8/1000 # 3.8mJ
freq = 1.

import pickle

# load face trace
with open("poi_1.pcl", "rb") as f:
    trace = pickle.load(f)

#print(trace)
def check_server(x, RTT, latency_limit):
    return x.s_loading_latency + x.s_compute_latency + RTT <= latency_limit

def check_server_cost(x, cost_budget, remaining_time, server_cost):
    return (cost_budget / remaining_time / freq) >= (x.s_compute_latency * server_cost)

def check_device_cost(x, energy_budget, remaining_time, is_away, time_from_last_swap):
    e_b = energy_budget
    if is_away:
        e_b -= x.loading_energy * (remaining_time / float(time_from_last_swap))
    return (e_b / remaining_time / freq) >= (x.compute_energy)

def check_device(x, latency_limit):
    return x.loading_latency + x.compute_latency + RTT < latency_limit

cnt = 0

current = -1

res = []
moves = 0
last_swapin = 0
prev_acc = 0
for i in trace:
    if i > 36000: break

    target_s = param.models
    target_c = param.models
    # server_side
    if current == -1: # cold miss
        target_s = filter(lambda x: check_server(x, RTT, latency_limit), target_s) 
        target_c = filter(lambda x: check_device(x, latency_limit), target_c) 

    target_s = filter(lambda x: check_server_cost(x, cost_budget, 36000-i, server_cost), target_s)
    target_c = filter(lambda x: check_device_cost(x, energy_budget, 36000-i, current==0, (i-last_swapin)), target_c)

    target_s.sort(key=lambda x:x.accuracy, reverse=True)
    target_c.sort(key=lambda x:x.accuracy, reverse=True)

    try: 
        server_pick = target_s[0]
    except:
        server_pick = None
    try:
        client_pick = target_c[0]
    except:
        client_pick = None
    if client_pick == None and server_pick == None:
        raise Exception()

    if client_pick == None or server_pick.accuracy > client_pick.accuracy:
        if current != 0:
            #print(i, "client->server")
            moves += 1
        current = 0  # server!!
        cost_budget -= server_pick.s_compute_latency * server_cost
        pick = server_pick
        energy_budget -= send_energy
    elif server_pick.accuracy == client_pick.accuracy and prev_acc == server_pick.accuracy:
        if current == 0:
            cost_budget -= server_pick.s_compute_latency * server_cost
            pick = server_pick
            energy_budget -= send_energy
        else:
            current = 1
            energy_budget -= client_pick.compute_energy
            pick = client_pick 
    else:
        if current != 1:
            #print(i, "server->client")
            moves += 1
            energy_budget -= client_pick.loading_energy
        current = 1  # client
        energy_budget -= client_pick.compute_energy
        pick = client_pick 

    res.append((i, energy_budget, cost_budget, pick.accuracy, current))
    prev_acc = pick.accuracy
#print(moves)
import matplotlib.pyplot as plt 
from mpltools import style
style.use('ggplot')
fig = plt.figure()

ax = fig.add_subplot(311)
ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
ax.set_ylabel('Energy Budget (J)')
ax2 = ax.twinx()
ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
ax2.set_ylabel('Cost Budget ($)')
ax.set_xlim(0,36000)
lns = ln + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=3)
ax = fig.add_subplot(312)
ln3 = ax.plot(map(lambda x:x[0], res), map(lambda x:x[3], res), label='Acc')
ax.set_ylabel('Accuracy (%)')
ax.set_xlim(0,36000)
ax = fig.add_subplot(313)
ln4 = ax.step(map(lambda x:x[0], res), map(lambda x:x[4], res), where='post', label='Acc')
ax.set_xlim(0,36000)
ax.set_ylim(-0.2, 1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(['server', 'client'])
#plt.show()

#fig.savefig('schedule_2wh_004.pdf', bbox_inches='tight')
fig.savefig('schedule1.pdf', bbox_inches='tight')


