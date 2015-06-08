import flycapture2 as fc2
import numpy as np
import cv2

def run():
    print fc2.get_library_version()
    c = fc2.Context()
    if (c.get_num_of_cameras() < 1): return
    c.connect(*c.get_camera_from_index(0))
    m, f = c.get_video_mode_and_frame_rate()
    #p = c.get_property(fc2.FRAME_RATE)
    c.start_capture()
    im = fc2.Image()
    while True:
        c.retrieve_buffer(im)
        a = np.array(im)
        cv2.imshow('rgb', a)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    c.stop_capture()
    c.disconnect()

if __name__ == "__main__":
    run()
