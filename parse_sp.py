#fields = ["sp_loading_energy", "sp_compute_energy", "sp_loading_latency", "sp_compute_latency", "sp_s_loading_latency", "sp_s_compute_latency"]
fields = ["name", "loading_energy", "compute_energy", "loading_latency", "compute_latency", "s_loading_latency", "s_compute_latency"]
with open("t", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        sp = line.split()
        print "models {"
        for i in range(len(sp)):
            if fields[i] in ["name"]: 
                print "   %s: \"%s\"" % (fields[i], sp[i])
            else:
                print "   %s: %s" % (fields[i], sp[i])
        print "}"
