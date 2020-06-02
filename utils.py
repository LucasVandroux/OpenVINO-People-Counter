from collections import namedtuple
import uuid

import cv2
import numpy as np

BoundingBox = namedtuple("BoundingBox", "x y w h")
# where (x,y) are the coordinates of the top-left corner of the bounding box 
# and w its width and h its height in pixels.

class PedestrianTracker:
    """ Pedestrian Tracker
    Class responsible to track each pedestrian in the frame individually.
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        """ Initialize the Pedestrian Tracker
        Args:
            x (int): x coordinate of the top-left corner of the inital bbox containing the pedestrian
            y (int): y coordinate of the top-left corner of the inital bbox containing the pedestrian
            w (int): width of the inital bbox containing the pedestrian
            h (int): height of the inital bbox containing the pedestrian
        """
        # Initialize the list of centroids of bboxes
        self.list_centroids = []
        # Update the current bbox and the list of centroids
        self.updateBbox(x, y, w, h)
        # Define a unique id for this tracker by taking the first 8 character of the uuid
        self.id = str(uuid.uuid1())[:8]

    def computeIouWith(self, bbox, min_iou=0.3):
        """ Compute the IoU
        Compute the Intersection Over Union (IoU) between the current bounding box containing the 
        pedestrian and another bounding box.
        Args:
            bbox ((x: int, y:int, w:int, h:int)): Bounding box to calculate the IoU with the 
                current bbox of the pedestrian. A bounding boxe is a tuple (x, y, w, h) where (x,y) 
                are the coordinates of the top-left corner of the bounding box and w its width and 
                h its height in pixels.
            min_iou (int: 0.3): minimum under which the value of the IoU will be set to 0.
        Returns:
            iou (float): value of the Intersection over Union between the current bbox of the pedestrian and the bbox given as an input.
        """
        # Extract bbox parameters from the input
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        bbox_area = bbox_w * bbox_h

        # find coordinates of the intersection
        xx1 = np.maximum(self.bbox.x, bbox_x)
        yy1 = np.maximum(self.bbox.y, bbox_y)
        xx2 = np.minimum(self.bbox.x + self.bbox.w, bbox_x + bbox_w)
        yy2 = np.minimum(self.bbox.y + self.bbox.h, bbox_y + bbox_h)

        # compute area of the intersection
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        # compute the IoU
        iou = intersection / (bbox_area + (self.bbox.w * self.bbox.h) - intersection)

        # set the iou to zero if it is inferior to a certain threshold.
        if iou < min_iou:
            iou = 0

        return iou

    def updateBbox(self, x: int, y: int, w: int, h: int):
        """ Update the current Bounding Box of the tracker and the list of centroids
        Args:
            x (int): x coordinate of the top-left corner of the bbox
            y (int): y coordinate of the top-left corner of the bbox
            w (int): width of the bbox
            h (int): height of the inital bbox
        """
        self.bbox = BoundingBox(x, y, w, h)
        self.list_centroids.append(self.computeCentroid(self.bbox))

    def computeCentroid(self, bbox):
        """ Compute the center of a bounding box
        Args:
            bbox (BoundingBox): bounding box to compute the center of.
        Returns:
            (centroid_x, centroid_y) ((int, int)): tuple containing the coordinates of the centroid
        """
        centroid_x = int(bbox.x + (bbox.w / 2))
        centroid_y = int(bbox.y + (bbox.h / 2))
        return (centroid_x, centroid_y)

    def drawOnFrame(self, frame, color=(0, 255, 0)):
        """ Draw the bbox and the previous centroids on an image
        Args:
            frame (np.array): image to draw the bbox and the previous centroids on.
            color ((B:int, G:int, R:int)): BGR values for the color to draw with.
        """
        # Draw the centroids
        for idx, centroid in enumerate(self.list_centroids):
            cv2.circle(frame, centroid, 2, color, -1)

        # Draw the bounding box
        cv2.rectangle(
            frame,
            (self.bbox.x, self.bbox.y),
            (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h),
            color,
            2,
        )

def updatePedestrians(list_pedestrians, list_detections):
    """ Update the pedestrians with a new bounding box if it matches a new detection
    Args:
        list_pedestrians (list[PedestrianTracker]): list of PedestrianTracker objects
        list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes. A bounding 
            boxe is a tuple (x, y, w, h) where (x,y) are the coordinates of the top-left corner of  
            the bounding box and w its width and h its height in pixels.
    Returns:
        list_pedestrians (list[PedestrianTracker]): list of PedestrianTracker that could be matched with a 
            new boudning box. Return the input list_pedestrians if no matching is needed.
        list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes that haven't 
            been matched to a pedestrian.
    """
    # If there are no detection then there is nothing to track
    if not list_detections:
        return [], list_detections

    # Initialize the array to store the Intersection Over Union (IoU) between the bounding boxes
    # of the pedestrians and the new detections.
    iou_pedestrians_detections = np.zeros([len(list_pedestrians), len(list_detections)])

    # Fill up the array with all the IoU values (row -> pedestrians, column -> detections)
    for idx, pedestrian in enumerate(list_pedestrians):
        for jdx, detection in enumerate(list_detections):
            iou_pedestrians_detections[idx, jdx] = pedestrian.computeIouWith(detection)

    # If the numpy array is not empty
    if iou_pedestrians_detections.size:
        # Initialize the list to keep track of detections that have been matched to a pedestrian
        list_index_detections_tracked = []
        # Initialize the list to keep track of the pedestrian that have been updated
        list_pedestrians_to_keep = []

        # Loop over the array until every pedestrian has been matched
        while np.sum(iou_pedestrians_detections):
            # Find coordinates of the max value in the array
            idx_pedestrian, idx_detection = np.unravel_index(
                np.argmax(iou_pedestrians_detections), iou_pedestrians_detections.shape
            )

            # Update the pedestrian bbox
            bbox_x, bbox_y, bbox_w, bbox_h = list_detections[idx_detection]
            list_pedestrians[idx_pedestrian].updateBbox(bbox_x, bbox_y, bbox_w, bbox_h)
            list_pedestrians_to_keep.append(list_pedestrians[idx_pedestrian])

            # Update the list with the detection that have been attributed to a pedestrian
            list_index_detections_tracked.append(idx_detection)

            # Set to zero the row of the pedestrian and the column of the detection
            iou_pedestrians_detections[idx_pedestrian, :] = 0
            iou_pedestrians_detections[:, idx_detection] = 0

        # Remove the detections that have been matched to a pedestrian already
        list_detections = [
            detection
            for idx, detection in enumerate(list_detections)
            if idx not in list_index_detections_tracked
        ]
        list_pedestrians = list_pedestrians_to_keep

    return list_pedestrians, list_detections