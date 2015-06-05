import struct
import math

class Format(object):
    """Endianness and size format for structures."""
    Native          = "@"       # Native format, native size
    StandardNative  = "="       # Native format, standard size
    LittleEndian    = "<"       # Standard size
    BigEndian       = ">"       # Standard size
    
class Element(object):
    """A single element in a struct."""
    id=0
    def __init__(self, typecode):
        Element.id+=1           # Note: not thread safe
        self.id = Element.id
        self.typecode = typecode
        self.size = struct.calcsize(typecode)

    def __len__(self):
        return self.size

    def decode(self, format, s):
        """Additional decode steps once converted via struct.unpack"""
        return s

    def encode(self, format, val):
        """Additional encode steps to allow packing with struct.pack"""
        return val

    def __str__(self):
        return self.typecode

    def __call__(self, num):
        """Define this as an array of elements."""
        # Special case - strings already handled as one blob.
        if self.typecode in 'sp':
            # Strings handled specially - only one item
            return Element('%ds' % num)
        else:
            return ArrayElement(self, num)

    def __getitem__(self, num): return self(num)

class ArrayElement(Element):
    def __init__(self, basic_element, num):
        Element.__init__(self, '%ds' % (len(basic_element) * num))
        self.num = num
        self.basic_element = basic_element

    def decode(self, format, s):
        # NB. We use typecode * size, not %s%s' % (size, typecode), 
        # so we deal with typecodes that already have numbers,  
        # ie 2*'4s' != '24s'
        return [self.basic_element.decode(format, x) for x in  
                    struct.unpack('%s%s' % (format, 
                            self.num * self.basic_element.typecode),s)]

    def encode(self, format, vals):
        fmt = format + (self.basic_element.typecode * self.num)
        return struct.pack(fmt, *[self.basic_element.encode(format,v) 
                                  for v in vals])

class EmbeddedStructElement(Element):
    def __init__(self, structure):
        Element.__init__(self, '%ds' % structure._struct_size)
        self.struct = structure

    # Note: Structs use their own endianness format, not their parent's
    def decode(self, format, s):
        return self.struct(s)

    def encode(self, format, s):
        return self.struct._pack(s)

name_to_code = {
    'Char'             : 'c',
    'Byte'             : 'b',
    'UnsignedByte'     : 'B',
    'Int'              : 'i',
    'UnsignedInt'      : 'I',
    'Short'            : 'h',
    'UnsignedShort'    : 'H',
    'Long'             : 'l',
    'UnsignedLong'     : 'L',
    'String'           : 's',  
    'PascalString'     : 'p',  
    'Pointer'          : 'P',
    'Float'            : 'f',
    'Double'           : 'd',
    'LongLong'         : 'q',
    'UnsignedLongLong' : 'Q',
    }

class Type(object):
    def __getattr__(self, name):
        return Element(name_to_code[name])

    def Struct(self, struct):
        return EmbeddedStructElement(struct)
        
Type=Type()

class MetaStruct(type):
    def __init__(cls, name, bases, d):
        type.__init__(cls, name, bases, d)
        if hasattr(cls, '_struct_data'):  # Allow extending by inheritance
            cls._struct_info = list(cls._struct_info) # use copy.
        else:
            cls._struct_data=''
            cls._struct_info=[]     # name / element pairs

        # Get each Element field, sorted by id.
        elems = sorted(((k,v) for (k,v) in d.iteritems() 
                        if isinstance(v, Element)),
                        key=lambda x:x[1].id)

        cls._struct_data += ''.join(str(v) for (k,v) in elems)
        cls._struct_info += elems
        cls._struct_size = struct.calcsize(cls._format + cls._struct_data)

class Struct(object):
    """Represent a binary structure."""
    __metaclass__=MetaStruct
    _format = Format.Native  # Default to native format, native size

    def __init__(self, _data=None, **kwargs):
        if _data is None:
            _data ='\0' * self._struct_size
            
        fieldvals = zip(self._struct_info, struct.unpack(self._format + 
                                             self._struct_data, _data))
        for (name, elem), val in fieldvals:
            setattr(self, name, elem.decode(self._format, val))
        
        for k,v in kwargs.iteritems():
            setattr(self, k, v)

    def _pack(self):
        return struct.pack(self._format + self._struct_data, 
            *[elem.encode(self._format, getattr(self, name)) 
                for (name,elem) in self._struct_info])                

    def __str__(self):
        return self._pack()
    
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._pack())

class MSG_HEADER(Struct):
    MessageNumber = Type.UnsignedByte
    MessageID = Type.UnsignedByte
    MessageSize = Type.UnsignedShort

MLX621_IR_ROWS = 4
MLX621_IR_COLUMNS = 16
MLX621_IR_SENSORS = MLX621_IR_ROWS * MLX621_IR_COLUMNS

class Sensor_EEPROM_t(Struct):
    delta_A = Type.UnsignedByte[MLX621_IR_SENSORS]
    Bi = Type.Byte[MLX621_IR_SENSORS]
    delta_Alpha = Type.UnsignedByte[MLX621_IR_SENSORS]
    reserved_C0_CF = Type.UnsignedByte[16]
    A_common = Type.Short
    KT_scale = Type.UnsignedByte
    A_CP_L = Type.UnsignedByte
    A_CP_H = Type.UnsignedByte
    B_CP = Type.Byte
    Alpha_CP = Type.UnsignedShort
    TGC = Type.Byte
    delta_A_B_scale = Type.UnsignedByte
    V_th = Type.Short
    K_T1 = Type.Short
    K_T2 = Type.Short
    Alpha0 = Type.UnsignedShort
    Alpha0_scale = Type.UnsignedByte
    delta_Alpha_scale = Type.UnsignedByte
    Epsilon = Type.UnsignedShort
    KsTa = Type.Short
    reserved_E8_EF = Type.UnsignedByte[8]
    reserved_F0_F4 = Type.UnsignedByte[5]
    CFG_L = Type.UnsignedByte
    CFG_H = Type.UnsignedByte
    OSC_Trim = Type.UnsignedByte
    ChipID = Type.UnsignedByte[8]

MLX621_CONFIG_REFRESH_RATE_512Hz = 0x5
ACTUAL_REFRESH_RATE = 0xA
ACTUAL_RESOLUTION = 3
MLX621_IR_ROWS = 4
MLX621_IR_COLUMNS = 16
MLX621_IR_SENSORS = MLX621_IR_ROWS * MLX621_IR_COLUMNS
SENSORS_ROWS = 6
SENSORS_COLS = 2
SENSORS_TOTAL = SENSORS_ROWS * SENSORS_COLS

import ctypes

class SensorInfo:
    def __init__(self, sensorID, row, column, eeprom):
        self.sensorID = sensorID
        self.row = row
        self.column = column
        self.eeprom = eeprom
        self.Ai = [0.0] * MLX621_IR_SENSORS
        self.Bi = [0.0] * MLX621_IR_SENSORS
        self.Alphai = [0.0] * MLX621_IR_SENSORS
        self.newAlphai = [0.0] * MLX621_IR_SENSORS

    def calc_temperature(self):
        AmbientTemp = self.Ta_ConstA * float(self.LastResults[0x40]) + self.Ta_ConstB
        if AmbientTemp >= 0: AmbientTemp = math.sqrt(AmbientTemp) + self.Ta_ConstC

        ambient_temp_minus25 = AmbientTemp - 25.0
        ambient_temp_power4 = math.pow(AmbientTemp + 273.15, 4)
        mystery_factor = 1.0 / (1.0 + self.KsTa * ambient_temp_minus25)

        tgc_compensation = float(self.LastResults[0x41]) - (self.ACP + self.BCP * ambient_temp_minus25)
        #print(self.ACP, self.BCP, ambient_temp_minus25, tgc_compensation)
        self.CompensationPixelTemp = tgc_compensation
        tgc_compensation *= self.TGC

        first_col = (self.sensorID % SENSORS_COLS) * MLX621_IR_COLUMNS
        first_row = (self.sensorID / SENSORS_COLS) * MLX621_IR_ROWS

        c = first_col
        self.Temperature = []
        i = 0
        while c < first_col + MLX621_IR_COLUMNS:
            r = first_row
            while r < first_row + MLX621_IR_ROWS:
                temp = float(self.LastResults[i]) - (self.Ai[i] + self.Bi[i] * ambient_temp_minus25)
                temp -= tgc_compensation
                temp *= (self.newAlphai[i] * mystery_factor)
                temp += ambient_temp_power4

                temp = math.pow(temp, 0.25) if (temp >= 0) else 0
                temp -= 273.15
                self.Temperature.append(temp)
                r += 1
                i += 1
            c += 1

    def calc_config(self):
        e = Sensor_EEPROM_t(self.eeprom)
        self.ConfigurationRegister = (e.CFG_H << 8) | e.CFG_L
        self.OSC_Trim = e.OSC_Trim
        self.IRRefreshRate = 512 / (1 << (ACTUAL_REFRESH_RATE - MLX621_CONFIG_REFRESH_RATE_512Hz))
        self.RefreshDelay = int(math.ceil(1000.0 / self.IRRefreshRate))

        self.SensorResolutionScale = 3 - ACTUAL_RESOLUTION
        KT1_Scale = ((e.KT_scale >> 4) & 0xF) + self.SensorResolutionScale
        KT2_Scale = ((e.KT_scale) & 0xF) + 10 + self.SensorResolutionScale
        Ai_Scale = (e.delta_A_B_scale >> 4) & 0xF
        Bi_Scale = e.delta_A_B_scale & 0xF

        self.Vtho = 1 if e.V_th == 0 else e.V_th
        self.Vtho /= math.pow(2.0, self.SensorResolutionScale)

        self.KT1 = float(e.K_T1) / self.Vtho / math.pow(2.0, KT1_Scale)
        self.KT2 = float(e.K_T2) / self.Vtho / math.pow(2.0, KT2_Scale)
        self.KsTa = float(e.KsTa) / math.pow(2.0, 20)

        self.Ta_ConstA = 1.0 / (self.KT2 * self.Vtho)
        self.Ta_ConstB = math.pow(0.5 * self.KT1 / self.KT2, 2) - 1.0 / self.KT2
        self.Ta_ConstC = 25.0 - 0.5 * self.KT1 / self.KT2

        self.TGC = float(e.TGC)/32.0 if e.TGC != 0 else 0.75
        temp = (e.A_CP_H << 8) | e.A_CP_L
        temp = ctypes.c_short(temp).value

        self.ACP = float(temp) / math.pow(2.0, self.SensorResolutionScale)
        self.BCP = float(e.B_CP) / math.pow(2.0, Bi_Scale + self.SensorResolutionScale)
        self.EmissivityCoefficient = 1.0 if e.Epsilon == 0 else (32768.0 / e.Epsilon)
        self.AlphaCP = float(e.Alpha_CP) / math.pow(2.0, e.Alpha0_scale + self.SensorResolutionScale)
        self.newAlphaCP = self.TGC * self.AlphaCP
        
        self.Alpha0 = float(e.Alpha0) / math.pow(2.0, e.Alpha0_scale + self.SensorResolutionScale)
        self.delta_Alpha_scaling_factor = 1.0 / math.pow(2.0, e.delta_Alpha_scale + self.SensorResolutionScale)

        for i in range(MLX621_IR_SENSORS):
            self.Ai[i] = float(e.A_common) + float(e.delta_A[i]) * math.pow(2.0, Ai_Scale) / math.pow(2.0, self.SensorResolutionScale)
            self.Bi[i] = float(e.Bi[i]) / math.pow(2.0, Bi_Scale + self.SensorResolutionScale)
            self.Alphai[i] = self.Alpha0 + self.delta_Alpha_scaling_factor * float(e.delta_Alpha[i])
            self.newAlphai[i] = self.EmissivityCoefficient / (self.Alpha0 + self.delta_Alpha_scaling_factor * (float(e.delta_Alpha[i]) - self.newAlphaCP))
           
           
FETCH_ALL_LENGTH_WORDS = MLX621_IR_SENSORS + 1 + 1

class MSG_NEW_RAWDATA_T(Struct):
    MessageNumber = Type.UnsignedByte
    MessageID = Type.UnsignedByte
    MessageSize = Type.UnsignedShort
    RawData = Type.Short[SENSORS_TOTAL * FETCH_ALL_LENGTH_WORDS]

    def handle(self, thermal):
        thermal.new_raw_data(self)
   
class MSG_SENSOR_EEPROM_T(Struct):
    MessageNumber = Type.UnsignedByte
    MessageID = Type.UnsignedByte
    MessageSize = Type.UnsignedShort
    
    SensorID = Type.UnsignedByte
    SensorRow = Type.UnsignedByte
    SensorCol = Type.UnsignedByte
    Reserved = Type.UnsignedByte
    EEPROM = Type.UnsignedByte[256]

    def handle(self, thermal):
        thermal.sensors[self.SensorID] = SensorInfo(self.SensorID, self.SensorRow, self.SensorCol, self.__str__()[-256:])
        thermal.sensors[self.SensorID].calc_config()

class MSG_SET_PARAMETERS_S(Struct):
    MessageNumber = Type.UnsignedByte
    MessageID = Type.UnsignedByte
    MessageSize = Type.UnsignedShort

    doSendRawThermalData = Type.UnsignedByte
    doSendProcessedThermalData = Type.UnsignedByte
    doSendIMUData = Type.UnsignedByte
    Sync_DecimationRate = Type.UnsignedByte

    Sync_TimeOut = Type.UnsignedShort
    Thermal_Refresh_Rate = Type.UnsignedByte
    Thermal_REsolution = Type.UnsignedByte

    Time_Year = Type.UnsignedShort
    Time_Month = Type.UnsignedByte
    Time_Day = Type.UnsignedByte

    Time_Hour = Type.UnsignedByte
    Time_Minute = Type.UnsignedByte
    Time_Seconds = Type.UnsignedByte
    Time_mSeconds = Type.UnsignedByte

    IMU_Command_Char = Type.UnsignedByte
    Reserved1 = Type.UnsignedByte
    Reserved2 = Type.UnsignedByte
    Reserved3 = Type.UnsignedByte



#m = MSG_SET_PARAMETERS_S()
#print(len(m._pack()))
