import cv2
from brainframe.api import BrainFrameAPI
import numpy as np
 
 
def read_frame(stream_uri, frame_index):
    cap = cv2.VideoCapture(stream_uri)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    rst, frame = cap.read()
    if not rst:
        print(f"Failed to read frame: {frame_index}")
    cap.release()
    return frame
 
 
def detect_image(api, frame, capsule_names=None):
    if capsule_names is None:
        capsule_names = ["object_detector"]
    detections = api.process_image(frame, capsule_names, {})
    return detections
 
def detect_object(detections, frame, output_path):
    # label object
    for detected_obj in detections:
        coords = detected_obj.coords
        class_name = detected_obj.class_name

        cv2.rectangle(frame, (coords[0][0], coords[0][1]), (coords[2][0], coords[2][1]), (0,0,255),1)
        
        t_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        ptLeftTop = np.array(coords[0])
        textlbottom =  ptLeftTop+ np.array(list(t_size))
        cv2.rectangle(frame, tuple(ptLeftTop), tuple(textlbottom),  (0, 0, 255), -1)
        ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
        cv2.putText(frame, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 1)

    # output
    cv2.imwrite(output_path+'result.png',frame)
    
    print("Output successfully")

def main():
    # The capsules for person and face detection
    capsule_names = ["object_detector"]
 
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    stream_path = "./videos/London_walk.mp4"

    # The output images
    output_path = './output/object/'


    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
 
    api = BrainFrameAPI(bf_server_url)
    api.wait_for_server_initialization()
 
    frame = read_frame(stream_path, 5055)
    if frame is None:
        return
 
    detections = detect_image(api, frame, capsule_names)

    # print(detections)

    # label objects and output images
    detect_object(detections, frame, output_path)

if __name__ == "__main__":
    main()