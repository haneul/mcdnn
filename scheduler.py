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

def check_split_cost(x, energy_budget, cost_budget, remaining_time, is_away, time_from_last_swap, freq):
    e_b = energy_budget
    if is_away:
        e_b -= x.sp_loading_energy * (remaining_time / float(time_from_last_swap))
    r1 = (e_b / remaining_time / freq) >= (x.sp_compute_energy)
    r2 = (cost_budget / remaining_time / freq) >= (x.sp_s_compute_latency * server_cost)
    return r1 and r2

def check_device(x, latency_limit):
    return x.loading_latency + x.compute_latency + RTT < latency_limit

def check_split(x, RTT, latency_limit):
    return max(x.sp_loading_latency, x.sp_s_loading_latency) + x.sp_compute_latency + x.sp_s_compute_latency + RTT < latency_limit


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
        self.last_swapin_split = 0
        self.res = []

import bisect 

class Scheduler:
    def __init__(self, name, energy_budget, cost_budget):
        self.name = name
        self.energy_budget = float(energy_budget)
        self.cost_budget = float(cost_budget)
        self.applications = {} 
        self.res = []
        self.connectivity = [(0,True)]
        self.connectivity_times = [0]
        self.use_split = False

    def add_application(self, app_type, application):
        self.applications[app_type] = application

    def set_connectivity(self, conn):
        self.connectivity = conn
        self.connectivity_times = map(lambda x:x[0], conn)

    def get_connectivity(self, i):
        j = bisect.bisect_left(self.connectivity_times, i)
        return self.connectivity[j-1][1]

    def rununtil(self, trace, until=36000):
        moves = 0
        prev_acc = 0
        for cur in trace:
            tApp = self.applications[cur[1]]
            i = cur[0]
            if i > until: break

            if self.get_connectivity(i):
                target_s = tApp.models
            else:
                target_s = []
            target_c = tApp.models
            if self.use_split:
                target_sp = tApp.models
            # server_side
            if tApp.status == Location.NOTRUNNING: # cold miss
                target_s = filter(lambda x: check_server(x, RTT, latency_limit), target_s) 
                target_c = filter(lambda x: check_device(x, latency_limit), target_c) 
                if self.use_split:
                    target_sp = filter(lambda x: check_split(x, RTT, latency_limit), target_sp)

            target_s = filter(lambda x: check_server_cost(x, self.cost_budget, until-i, server_cost, tApp.freq), target_s)
            target_c = filter(lambda x: check_device_cost(x, self.energy_budget, until-i, tApp.status==Location.SERVER, 
                (i-tApp.last_swapin), tApp.freq), target_c)
            if self.use_split:
                target_sp = filter(lambda x: check_split_cost(x, self.energy_budget, self.cost_budget, until-i, 
                    tApp.status==Location.SERVER or tApp.status==Location.DEVICE, (i-tApp.last_swapin_split), tApp.freq), target_sp)

            target_s.sort(key=lambda x:x.accuracy, reverse=True)
            target_c.sort(key=lambda x:x.accuracy, reverse=True)
            if self.use_split:
                target_sp.sort(key=lambda x:x.accuracy, reverse=True)

            picks = []
            try: 
                server_pick = target_s[0]
                server_pick.location = Location.SERVER
                picks.append(server_pick)
            except:
                server_pick = None
            try:
                client_pick = target_c[0]
                client_pick.location = Location.DEVICE
                picks.append(client_pick)
            except:
                client_pick = None
            if self.use_split:
                try:
                    split_pick = target_sp[0]
                    split_pick.location = Location.SPLIT
                    picks.append(split_pick)
                except:
                    split_pick = None


            #if client_pick == None and server_pick == None and split_pick == None:
            #print(picks)
            if len(picks) == 0:
                tApp.status = Location.NOTRUNNING
                tApp.res.append((i, 0, tApp.status))
                continue

            picks.sort(key=lambda x:x.accuracy, reverse=True)
            # TODO: tie break
            pick = picks[0]
            if pick.location == Location.SERVER:
                self.cost_budget -= server_pick.s_compute_latency * server_cost
                self.energy_budget -= send_energy

            elif pick.location == Location.DEVICE:
                if tApp.status != pick.location:
                    self.energy_budget -= client_pick.loading_energy
                    tApp.last_swapin = i
                self.energy_budget -= client_pick.compute_energy

            elif pick.location == Location.SPLIT:
                if tApp.status != pick.location:
                    self.energy_budget -= client_pick.sp_loading_energy
                    tApp.last_swapin_split = i
                self.energy_budget -= client_pick.sp_compute_energy
                self.cost_budget -= server_pick.sp_s_compute_latency * server_cost
                self.energy_budget -= send_energy

            else:
                print("ERROR")
                exit()

            tApp.status = pick.location

            """ 
            if server_pick != None and (client_pick == None or server_pick.accuracy > client_pick.accuracy):
                if tApp.status != Location.SERVER:
                    #print(i, "client->server")
                    moves += 1
                    tApp.status = Location.SERVER  # server!!
                self.cost_budget -= server_pick.s_compute_latency * server_cost
                pick = server_pick
                self.energy_budget -= send_energy
            elif server_pick != None and (server_pick.accuracy == client_pick.accuracy and prev_acc == server_pick.accuracy):
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
            """

            tApp.res.append((i, pick.accuracy, tApp.status))
            #print(i, self.energy_budget, self.cost_budget)
            self.res.append((i, self.energy_budget, self.cost_budget))
            prev_acc = pick.accuracy

param = model_pb2.ApplicationModel()
with open("model_sample.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)

app1 = Application("deepface", 1., param.models)

param2 = model_pb2.ApplicationModel()
with open("model_as.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param2)

app2 = Application("object-alex", .2, param2.models)

# sheculder 2
"""
scheduler = Scheduler("test1", 5*3600, 0.0667)
scheduler.add_application(AppType.FACE, app1)
scheduler.add_application(AppType.SCENE, app2)
"""

# split
#scheduler.add_application(AppType.SCENE, app2)

import pickle
# connectivity test
"""
with open("disconnect.pcl", "rb") as f:
    conn = pickle.load(f)
    scheduler.set_connectivity(conn)
"""
#energy_budget = 2.*3600 # 5Wh
#cost_budget = 0.04 # dollar ($2/month)


# load face trace
with open("poi_1.pcl", "rb") as f:
    trace = pickle.load(f)

with open("poi_5.pcl", "rb") as f:
    trace_5 = pickle.load(f)

#trace_5 = map(lambda x:(x, AppType.SCENE), trace_5)
#trace = trace + trace_5
#trace.sort(key=lambda x:x[0])


import matplotlib.pyplot as plt 
from mpltools import style
style.use('ggplot')

def depict(res, r1, filename):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
    ax.set_ylabel('Energy Budget (J)')
    ax.set_ylim(0,18000)
    ax2 = ax.twinx()
    ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
    ax2.set_ylabel('Cost Budget ($)')
    ax.set_xlim(0,36000)
    lns = ln + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=3)
    ax = fig.add_subplot(312)
    ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
    ax.set_xlim(0,36000)
    ax.set_ylabel('Accuracy (%)')
    ax = fig.add_subplot(313)
    ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
    ax.set_xlim(0,36000)
    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([3,2, 1])
    ax.set_yticklabels(['server','split', 'client'])
    plt.show()
    fig.savefig(filename, bbox_inches='tight')

def split():
    scheduler = Scheduler("split", 5*3600, 0.0667)
    scheduler.add_application(AppType.FACE, app1)
    scheduler.use_split = True
    trace = map(lambda x:(x, AppType.FACE), trace)

    scheduler.rununtil(trace)
    res = scheduler.res
    depict(res, app1.res, "split.pdf")

def special():
    param2 = model_pb2.ApplicationModel()
    with open("special_models.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param2)

special() 


"""
ax = fig.add_subplot(311)
ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
ax.set_ylabel('Energy Budget (J)')
ax.set_ylim(0,18000)
ax2 = ax.twinx()
ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
ax2.set_ylabel('Cost Budget ($)')
ax.set_xlim(0,36000)
lns = ln + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=3)
ax = fig.add_subplot(312)
r1 = app1.res
r2 = app2.res
ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
ln4 = ax.plot(map(lambda x:x[0], r2), map(lambda x:x[1], r2), label='App1')
ax.set_ylim(45,80)
ax.set_ylabel('Accuracy (%)')
ax.set_xlim(0,36000)
ax = fig.add_subplot(313)
ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
ln4 = ax.step(map(lambda x:x[0], r2), map(lambda x:x[2], r2), where='post', label='Acc')
ax.set_xlim(0,36000)
ax.set_ylim(0.8, 3.2)
ax.set_yticks([3, 1])
ax.set_yticklabels(['server', 'client'])
#plt.show()
ax = fig.add_subplot(411)
ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
ax.set_ylabel('Energy Budget (J)')
ax.set_ylim(0,18000)
ax2 = ax.twinx()
ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
ax2.set_ylabel('Cost Budget ($)')
ax.set_xlim(0,36000)
lns = ln + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=3)
ax = fig.add_subplot(412)
r1 = app1.res
r2 = app2.res
ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
ln4 = ax.plot(map(lambda x:x[0], r2), map(lambda x:x[1], r2), label='App1')
ax.set_ylim(45,80)
ax.set_ylabel('Accuracy (%)')
ax.set_xlim(0,36000)
ax = fig.add_subplot(413)
ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
ln4 = ax.step(map(lambda x:x[0], r2), map(lambda x:x[2], r2), where='post', label='Acc')
ax.set_xlim(0,36000)
ax.set_ylim(0.8, 3.2)
ax.set_yticks([3, 1])
ax.set_yticklabels(['server', 'client'])
ax = fig.add_subplot(414)
ln4 = ax.step(map(lambda x:x[0], conn), map(lambda x:x[1], conn), where='post', label='Acc')
ax.set_ylabel('Connectivity')
ax.set_xlim(0,36000)
ax.set_ylim(-0.2,1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Disconnected', 'Connected'])

#fig.savefig('schedule_2wh_004.pdf', bbox_inches='tight')
#fig.savefig('schedule2.pdf', bbox_inches='tight')
fig.savefig('schedule_disconn.pdf', bbox_inches='tight')
"""


