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

class Thermal:
    def __init__(self, handle):
        self.handle = handle
        self.sensors = {}
        self.Image = np.zeros((TotalColumns, TotalRows))#[[0]*TotalRows] * TotalColumns
        self.width = 640
        self.height = 480
        self.Im = create_blank(self.width, self.height)
        self.pixel_width = self.width / TotalColumns
        self.pixel_height = self.height / TotalRows
        self.color_map = load_colormap()
        self.len_cm = len(self.color_map)

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
        self.show_image()

    def show_image(self):
        for r in range(TotalRows):
            for c in range(TotalColumns):
                temp = self.Image[c][r]
                #print(temp)
                if temp < 20: temp = 20
                color_i = int((temp - 20) / 20.0 * self.len_cm)
                color = self.color_map[color_i]
                #print(self.pixel_width)
                x = c * self.pixel_width
                y = r * self.pixel_height
                cv2.rectangle(self.Im, (x, y), (x+self.pixel_width, y+self.pixel_height), color, -1)
        cv2.imshow('frame', self.Im)
        cv2.waitKey(1)
        

    def run(self):
        for i in range(600):    
            message = self._read_message()
            if len(message) == 0: continue
            header = MSG_HEADER(message[:4])
            msg = MsgFactory.create(header.MessageID, message)
            if msg != None:
                msg.handle(self)
            else:
                print(header.MessageID, header.MessageSize)


data = 0x00
handle.claimInterface(0)
try:
    res = handle.bulkWrite(0x01, m._pack(), 50)
except:
    pass

thermal = Thermal(handle)
thermal.run()
handle.releaseInterface(0)
handle.close()
cv2.waitKey()
cv2.destroyAllWindows()
