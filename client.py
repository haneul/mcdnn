import sys
import argparse
import xmlrpclib
import cmd

class DNNCmd(cmd.Cmd):
    def __init__(self, proxy, clientID):
        cmd.Cmd.__init__(self)
        self.proxy = proxy
        self.clientID = clientID
         
    def do_predict(self, line):
        cls = self.proxy.predict(self.clientID, line.strip())
        print(cls)

    def do_change(self, line):
        cmd = line.strip().split()
        res = self.proxy.change(self.clientID, cmd[0], cmd[1], bool(cmd[2]))
   
    def do_EOF(self, line):
        return True 
        

def main():
    parser = argparse.ArgumentParser(prog='mcdnn-client')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--digests', default=None, type=str)

    if len(sys.argv) < 2:
        sys.argv.append('--help')

    args = parser.parse_args()
    if(args.model == None or args.pretrained == None or args.digests == None):
        print("please specify model and pretrained file and digests")
        exit(0)

    proxy = xmlrpclib.ServerProxy("http://localhost:8000/")
    clientID = proxy.register(args.model, args.pretrained, args.digests, False)
    dnncmd = DNNCmd(proxy, clientID)
    dnncmd.cmdloop()
    proxy.unregister(clientID)
    
if __name__ == '__main__':
    main()
