import request_pb2
import struct
import socket
import time
import sys

def send_message(sock, message):
    s = message.SerializeToString()
    packed_len = struct.pack('>L', len(s))
    sock.sendall(packed_len + s)

def socket_read_n(sock, n):
    """ Read exactly n bytes from the socket.
        Raise RuntimeError if the connection closed before
        n bytes were read.
    """
    buf = ''
    while n > 0:
        data = sock.recv(n)
        if data == '':
            raise RuntimeError('unexpected connection close')
        buf += data
        n -= len(data)
    return buf

req = request_pb2.DNNRequest()
req.type = request_pb2.FACE

with open("../cat.jpg", "rb") as f:
    req.data = f.read()


#HOST, PORT = "archon.cs.washington.edu", 9999
HOST, PORT = "archon.cs.washington.edu", int(sys.argv[1])
# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    beg = time.time()
    send_message(sock, req)
    len_buf = socket_read_n(sock, 4)
    msg_len = struct.unpack('>L', len_buf)[0]
    msg_buf = socket_read_n(sock, msg_len)
    msg = request_pb2.DNNResponse() 
    msg.ParseFromString(msg_buf)
    end = time.time()
    print(msg.success)
    print(msg.result)
    print( (end-beg) * 1000 )
    
finally:
    sock.close()




