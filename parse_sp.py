fields = ["sp_loading_energy", "sp_compute_energy", "sp_loading_latency", "sp_compute_latency", "sp_s_loading_latency", "sp_s_compute_latency"]
with open("t", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        sp = line.split()
        for i in range(len(sp)):
            print "   ",
            print fields[i],
            print ": ",
            print sp[i]
        print
