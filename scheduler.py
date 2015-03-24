import model_pb2
import google.protobuf.text_format


#for model in param.models:
#    print(model)

latency_limit = 1000 # 2secs
RTT = 100 # 100ms
server_cost = 1.8e-7 # $ per ms
send_energy = 3.8/1000 # 3.8mJ

def check_server(x, RTT, latency_limit):
    return x.s_loading_latency + x.s_compute_latency + RTT <= latency_limit

def check_server_cost(x, cost_budget, remaining_time, server_cost, freq):
    return (cost_budget / remaining_time / freq) >= (x.s_compute_latency * server_cost)

def check_device_cost(x, energy_budget, remaining_time, is_away, time_from_last_swap, freq):
    e_b = energy_budget
    if is_away:
        e_b -= x.loading_energy * (remaining_time / float(time_from_last_swap))
    return (e_b / remaining_time / freq) >= (x.compute_energy)

def check_device(x, latency_limit):
    return x.loading_latency + x.compute_latency + RTT < latency_limit

class Location:
    NOTRUNNING, DEVICE, SPLIT, SERVER = range(0,4)

class AppType:
    FACE, OBJECT, SCENE = range(3)

class Application:
    def __init__(self, name, freq, models):
        self.name = name
        self.freq = float(freq)
        self.models = models
        self.status = Location.NOTRUNNING
        self.last_swapin = 0

import copy

class Scheduler:
    def __init__(self, name, energy_budget, cost_budget):
        self.name = name
        self.energy_budget = float(energy_budget)
        self.cost_budget = float(cost_budget)
        self.applications = {} 
        self.res = []

    def add_application(self, app_type, application):
        self.applications[app_type] = application

    def rununtil(self, trace, until=36000):
        moves = 0
        prev_acc = 0
        for cur in trace:
            tApp = self.applications[cur[1]]
            i = cur[0]
            if i > until: break

            target_s = tApp.models
            target_c = tApp.models
            # server_side
            if tApp.status == Location.NOTRUNNING: # cold miss
                target_s = filter(lambda x: check_server(x, RTT, latency_limit), target_s) 
                target_c = filter(lambda x: check_device(x, latency_limit), target_c) 

            target_s = filter(lambda x: check_server_cost(x, self.cost_budget, until-i, server_cost, tApp.freq), target_s)
            target_c = filter(lambda x: check_device_cost(x, self.energy_budget, until-i, tApp.status==Location.SERVER, 
                (i-tApp.last_swapin), tApp.freq), target_c)

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
                if tApp.status != Location.SERVER:
                    #print(i, "client->server")
                    moves += 1
                    tApp.status = Location.SERVER  # server!!
                self.cost_budget -= server_pick.s_compute_latency * server_cost
                pick = server_pick
                self.energy_budget -= send_energy
            elif server_pick.accuracy == client_pick.accuracy and prev_acc == server_pick.accuracy:
                if tApp.status == Location.SERVER:
                    self.cost_budget -= server_pick.s_compute_latency * server_cost
                    pick = server_pick
                    self.energy_budget -= send_energy
                else:
                    tApp.status = Location.DEVICE 
                    self.energy_budget -= client_pick.compute_energy
                    pick = client_pick 
            else:
                if tApp.status != Location.DEVICE:
                    #print(i, "server->client")
                    moves += 1
                    self.energy_budget -= client_pick.loading_energy
                    tApp.last_swapin = i
                tApp.status = Location.DEVICE  # client
                self.energy_budget -= client_pick.compute_energy
                pick = client_pick 

            self.res.append((i, self.energy_budget, self.cost_budget, pick.accuracy, tApp.status))
            prev_acc = pick.accuracy

param = model_pb2.ApplicationModel()
with open("model_sample.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)

app1 = Application("deepface", 1., param.models)
# sheculder 1
scheduler = Scheduler("test1", 5*3600, 0.0667)
scheduler.add_application(AppType.FACE, app1)

#energy_budget = 2.*3600 # 5Wh
#cost_budget = 0.04 # dollar ($2/month)

import pickle

# load face trace
with open("poi_1.pcl", "rb") as f:
    trace = pickle.load(f)

trace = map(lambda x:(x, AppType.FACE), trace)

scheduler.rununtil(trace)
res = scheduler.res

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
ax.set_ylim(0.8, 3.2)
ax.set_yticks([3, 1])
ax.set_yticklabels(['server', 'client'])
#plt.show()

#fig.savefig('schedule_2wh_004.pdf', bbox_inches='tight')
fig.savefig('schedule1.pdf', bbox_inches='tight')


