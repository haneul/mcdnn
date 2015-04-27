import threading
import os
import wget
from threading import Thread


cnt = 0
target_file = open("images.tsv")
lock = threading.Lock()

class myThread(Thread):
    def run(self):
        global cnt
        while True:
            lock.acquire()
            try:
                d = target_file.next()
            except:
                lock.release()
                break

            sp = d.strip().split()
            dirname = "down/%s" % sp[1] 
            try:
                os.makedirs(dirname)
            except:
                pass
            cnt += 1
            if cnt % 10000 == 0:
                print "downloading %d" % cnt
            lock.release()

            try:
                wget.download(sp[2], out=dirname, bar=None)
            except:
                print("error downloading %s" % sp[2])
            if(cnt >= 100): break

threads = []
for i in range(10):
    threads.append(myThread())
for th in threads:
    th.start()
for th in threads:
    th.join()

target_file.close()
