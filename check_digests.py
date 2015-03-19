import sys, pickle

with open(sys.argv[1]) as f:
    dic = pickle.load(f)
print(dic)

