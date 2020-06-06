from collections import namedtuple
import uuid

import cv2
import numpy as np

BoundingBox = namedtuple("BoundingBox", "x y w h")
# where (x,y) are the coordinates of the top-left corner of the bounding box 
# and w its width and h its height in pixels.

class Tracker:
    """ Pedestrian Tracker
    Class responsible to track each object in the frame individually.
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        """ Initialize the Pedestrian Tracker
        Args:
            x (int): x coordinate of the top-left corner of the inital bbox containing the tracked object
            y (int): y coordinate of the top-left corner of the inital bbox containing the tracked object
            w (int): width of the inital bbox containing the tracked object
            h (int): height of the inital bbox containing the tracked object
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
        object and another bounding box.
        Args:
            bbox ((x: int, y:int, w:int, h:int)): Bounding box to calculate the IoU with the 
                current bbox of the object. A bounding boxe is a tuple (x, y, w, h) where (x,y) 
                are the coordinates of the top-left corner of the bounding box and w its width and 
                h its height in pixels.
            min_iou (int: 0.3): minimum under which the value of the IoU will be set to 0.
        Returns:
            iou (float): value of the Intersection over Union between the current bbox of the object and the bbox given as an input.
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

    def drawOnFrame(self, frame, color=(0, 255, 0), lost_color=(128, 128, 128)):
        """ Draw the bbox and the previous centroids on an image
        Args:
            frame (np.array): image to draw the bbox and the previous centroids on.
            color ((B:int, G:int, R:int)): BGR values for the color to draw with.
            lost_color ((B:int, G:int, R:int)): BGR value for the color for the bbox is the tracker is lost
        """
        # Draw the centroids
        for idx, centroid in enumerate(self.list_centroids):
            if centroid is not None:
                cv2.circle(frame, centroid, 2, color, -1)

        bbox_color = lost_color if self.list_centroids[-1] is None else color

        # Draw the bounding box
        cv2.rectangle(
            frame,
            (self.bbox.x, self.bbox.y),
            (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h),
            bbox_color,
            2,
        )

    def lost(self):
        """ Mark the tracker as lost
        """
        self.list_centroids.append(None)
    
    def isLost(self):
        """ Return the number of frame the tracker was lost for

        Returns:
            (int): return the number of last consecutive frame a tracker as not detected in.
        """
        reversed_list_centroids = reversed(self.list_centroids)
        lost_count = 0

        for centroid in reversed_list_centroids:
            if centroid is None:
                lost_count += 1
            else:
                break

        return lost_count

def updateTrackers(list_trackers, list_detections, num_frames_keep_lost_tracker:int=1):
    """ Update the trackers with a new bounding box if it matches a new detection
    Args:
        list_trackers (list[Tracker]): list of Tracker objects
        list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes. A bounding 
            boxe is a tuple (x, y, w, h) where (x,y) are the coordinates of the top-left corner of  
            the bounding box and w its width and h its height in pixels.
        num_frames_keep_lost_tracker (int:1) Number of consecutive frames a lost tracker should be kept
    Returns:
        list_trackers_to_keep (list[Tracker]): list of Tracker that could be matched with a 
            new boudning box or that should still be kept.
        list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes that haven't 
            been matched to an tracker.
    """
    # Initialize the list to keep track of the tracker that have been updated
    list_trackers_to_keep = []
    
    # If there are no detection then there is nothing to track
    if not list_detections:
        # if no detection set all the trackers to lost
        for tracker in list_trackers:
            tracker.lost()

        list_trackers_to_keep = [tracker for tracker in list_trackers if tracker.isLost() <= num_frames_keep_lost_tracker]
        
    else: 
        # Initialize the array to store the Intersection Over Union (IoU) between the bounding boxes
        # of the trackers and the new detections.
        iou_trackers_detections = np.zeros([len(list_trackers), len(list_detections)])

        # Fill up the array with all the IoU values (row -> trackers, column -> detections)
        for idx, tracker in enumerate(list_trackers):
            for jdx, detection in enumerate(list_detections):
                iou_trackers_detections[idx, jdx] = tracker.computeIouWith(detection)

        # If the numpy array is not empty
        if iou_trackers_detections.size:
            # Initialize the list to keep track of detections that have been matched to a tracker
            list_index_detections_tracked = []
            # Initialize the list to keep track of the idx of the tracker not lost
            list_idx_trackers_to_keep = []

            # Loop over the array until every tracker has been matched
            while np.sum(iou_trackers_detections):
                # Find coordinates of the max value in the array
                idx_tracker, idx_detection = np.unravel_index(
                    np.argmax(iou_trackers_detections), iou_trackers_detections.shape
                )

                # Update the tracker bbox
                bbox_x, bbox_y, bbox_w, bbox_h = list_detections[idx_detection]
                list_trackers[idx_tracker].updateBbox(bbox_x, bbox_y, bbox_w, bbox_h)
                list_trackers_to_keep.append(list_trackers[idx_tracker])
                list_idx_trackers_to_keep.append(idx_tracker)

                # Update the list with the detection that have been attributed to a tracker
                list_index_detections_tracked.append(idx_detection)

                # Set to zero the row of the tracker and the column of the detection
                iou_trackers_detections[idx_tracker, :] = 0
                iou_trackers_detections[:, idx_detection] = 0

            # Remove the detections that have been matched to a tracker already
            list_detections = [
                detection
                for idx, detection in enumerate(list_detections)
                if idx not in list_index_detections_tracked
            ]
            # Take care of the lost trackers
            list_trackers_lost = [list_trackers[idx] for idx in range(len(list_trackers)) if idx not in list_idx_trackers_to_keep]
            for tracker in list_trackers_lost:
                tracker.lost()
            # Remove the trackers that have been lost for more than a certain number of frame
            list_trackers_lost_to_keep = [tracker for tracker in list_trackers_lost if tracker.isLost() <= num_frames_keep_lost_tracker]
            # Add lost trackers to keep to the final list of trackers
            list_trackers_to_keep += list_trackers_lost_to_keep

    return list_trackers_to_keep, list_detections