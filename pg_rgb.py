import flycapture2 as fc2
import numpy as np
import cv2


from threading import Thread

class ImageThread(Thread):
    def __init__(self, ctrl = None):
        Thread.__init__(self)
        self.ctrl = ctrl 

    def run(self):
        self._run()

    def _run(self, keep = True):
        print fc2.get_library_version()
        c = fc2.Context()
        if (c.get_num_of_cameras() < 1): return
        c.connect(*c.get_camera_from_index(0))
        m, f = c.get_video_mode_and_frame_rate()
        print(c.get_format7_configuration())
        c.start_capture()
        im = fc2.Image()
        while True:
            c.retrieve_buffer(im)
            a = np.array(im)
            a = cv2.resize(a, (640,480))
            a = cv2.flip(a, 0)
            if self.ctrl:
                self.ctrl.lock.acquire()
                self.ctrl.imageQueue.append(a)
                self.ctrl.haveImage = True
                self.ctrl.lock.release()
                self.ctrl.event() 
                #cv2.imshow('rgb1', a)
                #cv2.waitKey(1)
            else:
                cv2.imshow('rgb', a)

                if not keep:
                    key = cv2.waitKey()
                    break
                else:
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'): break
        c.stop_capture()
        c.disconnect()

if __name__ == "__main__":
    im = ImageThread()
    im._run(False)
