import cmd

class Hi(cmd.Cmd):
    def do_hoho(self, line):
        print line
    def do_haha(self, line):
        print "haha" + line
    def do_EOF(self, line):
        return True

Hi().cmdloop()
