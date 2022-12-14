# Import dependencies
import numpy as np
from typing import Dict

from vcap import (
    BaseCapsule,
    NodeDescription,
    DetectionNode,
    FloatOption,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    rect_to_coords,
)
from vcap_utils import TFObjectDetector

def get_labels():
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 
    'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush', 'hair brush'] 
    return classes

# Define the Capsule class
class Capsule(BaseCapsule):
    # Metadata of this capsule
    name = "object_detector"
    description = "This is an example of how to wrap a TensorFlow Object " \
                  "Detection API model"
    version = 1

    # Define the input type. Since this is an object detector, and doesn't
    # require any input from other capsules, the input type will be a 
    # NodeDescription with size=NONE.
    input_type = NodeDescription(size=NodeDescription.Size.NONE)

    # Define the output type. In this case, as we are going to return a list of
    # bounding boxes, the output type will be size=ALL. The type of detection
    # will be "face", and we will place the detection confidence in extra_data.
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=get_labels(),
        extra_data=["detection_confidence"]
    )

    # Define the backend_loader
    backend_loader = lambda capsule_files, device: Backend(
        device=device,
        model_bytes=capsule_files["detector_object_tf.pb"],
        metadata_bytes=capsule_files["dataset_metadata.json"])

    # The options for this capsule. In this example, we will allow the user to
    # set a threshold for the minimum detection confidence. This can be adjusted
    # using the BrainFrame client or through REST API.
    options = {
        "threshold": FloatOption(
            description="Filter out bad detections",
            default=0.5,
            min_val=0.0,
            max_val=1.0,
        )
    }

# Define the Backend Class
class Backend(TFObjectDetector):
    def process_frame(self, frame: np.ndarray,
                      detection_node: None,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        """
        :param frame: A numpy array of shape (height, width, 3)
        :param detection_node: None
        :param options: Example: {"threshold": 0.5}. Defined in Capsule class above.
        :param state: (Unused in this capsule)
        :return: A list of detections
        """

        # Send the frame to the BrainFrame backend. This function will return a
        # queue. BrainFrame will batch_process() received frames and populate
        # the queue with the results.
        prediction_output_queue = self.send_to_batch(frame)

        # Wait for predictions
        predictions = prediction_output_queue.result()

        # Iterate through all the predictions received in this frame
        detection_nodes = []
        for prediction in predictions:
            # Filter out detection with low confidence.
            if prediction.confidence < options["threshold"]:
                continue
            # Create a DetectionNode for the prediction. It will be reused by 
            # any other capsules that require a face DetectionNode in their 
            # input type. An age classifier capsule would be an example of such 
            # a capsule.
            new_detection = DetectionNode(
                name=prediction.name,
                # convert [x1, y1, x2, y2] to [[x1,y1], [x1, y2]...]
                coords=rect_to_coords(prediction.rect),
                extra_data={"detection_confidence": prediction.confidence}
            )
            detection_nodes.append(new_detection)

        return detection_nodes