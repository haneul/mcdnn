import usb1
from usbmsgs import *

USB_VID = 0x04b4
USB_PID = 0x8051

context = usb1.USBContext()
handle = context.openByVendorIDAndProductID( USB_VID, USB_PID, skip_on_error = True)
if handle is None:
    print("ERROR")
    exit(0)
print(handle)
IN = 0x80
OUT = 0x00
m = MSG_SET_PARAMETERS_S()
m.MessageNumber = 0
m.MessageID = 8
m.MessageSize = 20
m.doSendIMUData = 1
m.doSendProcessedThermalData = 0
m.Sync_DecimationRate = 1
m.Sync_TimeOut = 300
m.Thermal_Refresh_Rate = 32
m.Thermal_Resolution = 18
m.IMU_Command_Char = 0

class MsgFactory:
    msgdict = {6: MSG_SENSOR_EEPROM_T, 7: MSG_NEW_RAWDATA_T}

    @staticmethod
    def create(msg_id, message):
        if not MsgFactory.msgdict.has_key(msg_id):
            return None
        cls = MsgFactory.msgdict[msg_id]
        msg = cls(message)
        return msg
             
RowsPerSensor = 4
ColumnsPerSensor = 16
NumRowsOfSensor = 6
NumColumnsOfSensor = 2
TotalColumns = NumColumnsOfSensor * ColumnsPerSensor
TotalRows = NumRowsOfSensor * RowsPerSensor

import cv, cv2
import numpy as np
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def load_colormap():
    cmap = {}
    lst = []
    with open("CoolWarmUChar257.csv") as f:
        for line in f:
            line = line.strip()
            sp = line.split(",")
            sp = map(float, sp)
            lst.append((sp[3], sp[2], sp[1]))
    return lst 

from threading import Thread
class Thermal(Thread):
    def __init__(self, handle, ctrl):
        Thread.__init__(self)
        self.handle = handle
        self.sensors = {}
        self.Image = np.zeros((TotalColumns, TotalRows))#[[0]*TotalRows] * TotalColumns
        self.width = 640
        self.height = 480
        self.Im = create_blank(self.width, self.height)
        self.ctrl = ctrl

    def _read_message(self):
        message = ''
        while True:
            packet = self.handle.bulkRead(0x82, 64)
            message += packet
            if len(packet) % 64 == 0 and len(packet) != 0:
                continue
            else:
                break
        return message

    def new_raw_data(self, msg):
        for sensor_id in range(SENSORS_TOTAL):
            self.sensors[sensor_id].LastResults = [0] * FETCH_ALL_LENGTH_WORDS
            for i in range(FETCH_ALL_LENGTH_WORDS):
                self.sensors[sensor_id].LastResults[i] = msg.RawData[sensor_id * FETCH_ALL_LENGTH_WORDS + i]
            self.sensors[sensor_id].calc_temperature()

        scale = 1.0
        offset = 0.0
        self.Image = np.zeros((TotalColumns, TotalRows))#[[0]*TotalRows] * TotalColumns

        for sensor_id in range(SENSORS_TOTAL):
            i = 0
            first_col = sensor_id % NumColumnsOfSensor * ColumnsPerSensor;
            first_row = sensor_id / NumColumnsOfSensor * RowsPerSensor
            c = first_col
            while c < first_col + ColumnsPerSensor:
                r = first_row
                while r < first_row + RowsPerSensor:
                    self.Image[c][r] = self.sensors[sensor_id].Temperature[i] * scale + offset
                    r += 1
                    i += 1
                c += 1
        self.ctrl.lock.acquire()
        self.ctrl.tempQueue.append(self.Image)
        self.ctrl.haveTemp = True
        self.ctrl.lock.release()
        self.ctrl.event()
        return True
        #return self.show_image()

    def show_image(self):
        for r in range(TotalRows):
            for c in range(TotalColumns):
                temp = self.Image[c][r]
                if temp < 20: temp = 20
                if temp > 40: temp = 39.9
                color_i = int((temp - 20) / 20.0 * self.len_cm)
                color = self.color_map[color_i]
                #print(self.pixel_width)
                x = c * self.pixel_width
                y = r * self.pixel_height
                cv2.rectangle(self.Im, (x, y), (x+self.pixel_width, y+self.pixel_height), color, -1)
        cv2.imshow('frame', self.Im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return False
        return True
       
    def run(self):
        while True:
            try:
                if not self._run_():
                    break
            except Exception as e:
                print("error", e)
                break
         

    def _run_(self):
        message = self._read_message()
        if len(message) == 0: return True
        header = MSG_HEADER(message[:4])
        msg = MsgFactory.create(header.MessageID, message)
        if msg != None:
            return msg.handle(self)
        else:
            #print(header.MessageID, header.MessageSize)
            return True

import Queue
from threading import Lock
from util import sendFrame
import request_pb2
HOST = "archon.cs.washington.edu"
PORT = 9999
label = ''

class SendThread(Thread):
    def __init__(self, frame):
        Thread.__init__(self)
        self.frame = frame

    def run(self):
        global label
        try:
            label_, latency = sendFrame(self.frame, HOST, PORT, request_pb2.OBJECT)
        except:
            return
        label = label_
        print(label)
        
import time
last = time.time()

class ShowThread(Thread):
    def __init__(self, im, temp):
        Thread.__init__(self)
        self.Image = im
        self.temp = temp
        self.width = 640
        self.height = 480
        self.temp_im = create_blank(self.width, self.height)
        self.pixel_width = self.width / TotalColumns
        self.pixel_height = self.height / TotalRows
        self.color_map = load_colormap()
        self.len_cm = len(self.color_map)

    def show_image(self):
        for r in range(TotalRows):
            for c in range(TotalColumns):
                temp = self.temp[c][r]
                if temp < 20: temp = 20
                if temp > 40: temp = 39.9
                color_i = int((temp - 20) / 20.0 * self.len_cm)
                color = self.color_map[color_i]
                #print(self.pixel_width)
                x = c * self.pixel_width
                y = r * self.pixel_height
                cv2.rectangle(self.temp_im, (x, y), (x+self.pixel_width, y+self.pixel_height), color, -1)
        cv2.imshow('frame', self.temp_im)
        return True

    def run(self):
        #self.show_image()
        temp = self.temp > 32
        shape = temp.shape
        rightmost = -1
        topmost = 100
        found = False
        for i in range(shape[0]):
            for j in range(shape[1]):
                if temp[i][j]:
                    if i > rightmost: rightmost = i
                    if j < topmost: topmost = j
                    found = True
        if found: 
            global last
            h = 300
            x = max(0, min(int(rightmost/32.0*640 - h/2), 640-h))
            y = max(0, min(int(topmost/24.0*480)-30, 480-h))
            cv2.rectangle(self.Image, (x, y), (x+h, y+h), (255,0,0), 3)
            now = time.time()
            if now-last > 1:
                SendThread(self.Image[y:y+h, x:x+h]).start()
                last = now
        
        cv2.imshow('rgb', self.Image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return False


class GlimpseController:
    def __init__(self):
        self._reset()
        self.lock = Lock()

    def _reset(self):
        self.imageQueue = []
        self.tempQueue = [] 
        self.haveImage = False
        self.haveTemp = False

    def event(self):
        self.lock.acquire()
        if self.haveImage and self.haveTemp:
            im = self.imageQueue[-1]
            temp = self.tempQueue[-1]
            self._reset()
            self.lock.release() 
            st = ShowThread(im, temp)
            st.start()
        else: 
            self.lock.release() 


ctrl = GlimpseController()

data = 0x00
handle.claimInterface(0)
try:
    res = handle.bulkWrite(0x01, m._pack(), 50)
except:
    pass

thermal = Thermal(handle, ctrl)
thermal.start()

from pg_rgb import ImageThread
im = ImageThread(ctrl)
im.start()

thermal.join()
im.join()
handle.releaseInterface(0)
handle.close()
#cv2.waitKey()
cv2.destroyAllWindows()
