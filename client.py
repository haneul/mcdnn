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
   
    def do_EOF(self, line):
        return True 
        

def main():
    parser = argparse.ArgumentParser(prog='mcdnn-client')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--pretrained', default=None, type=str)

    if len(sys.argv) < 2:
        sys.argv.append('--help')

    args = parser.parse_args()
    if(args.model == None or args.pretrained == None):
        print("please specify model and pretrained file")
        exit(0)

    proxy = xmlrpclib.ServerProxy("http://localhost:8000/")
    clientID = proxy.register(args.model, args.pretrained)
    dnncmd = DNNCmd(proxy, clientID)
    dnncmd.cmdloop()
    proxy.unregister(clientID)
    
if __name__ == '__main__':
    main()
