import numpy as np

from kalman_filter import KalmanFilter
from track import Track, TrackState
from assignment import cost_matrix, linear_assignment

from collections import defaultdict

CHI2_INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

BIG_NUM = 1e6

class Tracker(object):
    """
    SORT-based Tracker implementation with per-label tracking and adaptive thresholding.
    General use is to pass detections one frame at a time. 
    """
    def __init__(self, keep_tracks=False):
        # Shared objects for managing Tracks
        self.kalman_filter = KalmanFilter()
        self.curr_tracks = {} # track_id --> Track
        
        # Adaptive Threshold/Variables
        self.score_thresh = defaultdict(lambda: 0.01) # label --> score_thresh (float)
                
        # Counter variables 
        self.curr_track = 0
        self.curr_frame = 0
        
        # Used for storing global track information
        if keep_tracks:
            self.all_tracks = {} # track_id --> Track
    
    def update_score_thresh(
            self, 
            scores: np.ndarray, 
            labels: np.ndarray
        ):
        """
        Updates the per-label score threshold dynamically. 

        Args:
            scores : (N,) ndarray
                the scores of the current frame of a given label
            label : int 
                the label for which we wish to update the threshold for
        """
        unique_labels = np.unique(labels)
        for label in unique_labels:
            filt_scores = scores[(labels == label)]
            
            if len(filt_scores) == 0:
                continue
            
            new_thresh = np.max(np.abs(np.sort(filt_scores))) - 0.01
            if len(filt_scores) == 1:
                new_thresh = max(0.01, filt_scores[0] - 0.01)
            
            if self.score_thresh[label] == 0.01:
                self.score_thresh[label] = new_thresh
            
            self.score_thresh[label] = 0.9 * self.score_thresh[label] + 0.1 * new_thresh
        
    def update(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        labels: np.ndarray
    ):
        """
        the function to be called every frame

        Args:
            boxes : (N,4) np.ndarray
                the current frame bounding boxes (x, y, w, h) 
            scores : (N,) np.ndarray
                the current frame scores
            labels : (N,) np.ndarray
                the current frame labels
        """
        N = len(self.curr_tracks)   # Num Tracks
        M = len(boxes)              # Num Dets
        
        # if there are no dets for the current frame we end early
        if M == 0:
            self.curr_frame += 1
            return 
        
        # Process current tracks for matching
        for track in self.curr_tracks.values():
            # Forward tracks by 1 frame
            track.mean, track.covariance = self.kalman_filter.predict(track.mean, track.covariance)

        # Initialize the current tracks into a 2D matrix for computation
        curr_tracks = []
        curr_track_ids = []
        curr_track_labels = []
        gating_distances = []
        
        for track_id, track in self.curr_tracks.items():
            curr_track_ids.append(track_id)
            curr_tracks.append(track.tlbr)
            curr_track_labels.append(track.label)
            gating_distances.append(self.kalman_filter.gating_distance(track.mean, track.covariance, boxes))
        
        curr_track_ids = np.asarray(curr_track_ids)       # (N,) track ids for accessing tracks
        curr_track_labels = np.asarray(curr_track_labels) # (N,) track labels for per-label processing
        curr_tracks = np.vstack(curr_tracks)            # (N, 4) track boxes
        gating_distances = np.vstack(gating_distances)  # (N, M) gating distances^2
        
        # Update the dynamic threshold for all classes
        self.update_score_thresh(scores, labels)
        
        # Compute the cost matrix
        base_costs = cost_matrix(curr_tracks, boxes)

        # Variables for matches
        all_matches = []
        matched_track_mask = np.zeros(N, dtype=bool)
        matched_det_mask   = np.zeros(M, dtype=bool)
        
        # 1st Pass - Match high scoring dets
        for label in np.unique(curr_track_labels):
            # Masks for filtering by tracks, dets, and Mahalanobis distance^2 (dof=4)
            track_mask = (curr_track_labels == label) & (~matched_track_mask)
            det_mask = (labels == label) & (scores >= self.score_thresh[label]) & (~matched_det_mask)
            gating_mask = (gating_distances < CHI2_INV95[4])
            
            if not track_mask.any() or not det_mask.any() or not (gating_distances[track_mask][:, det_mask] < CHI2_INV95[4]).any():
                continue
            
            # Filter the cost matrix by the masks
            cost = base_costs.copy()
            cost[~track_mask, :] = BIG_NUM
            cost[:, ~det_mask]   = BIG_NUM
            cost[~gating_mask]   = BIG_NUM
            
            # Linear assignment of tracks & dets
            matches, unmatched_tracks, unmatched_dets = linear_assignment(cost, thresh=0.7)
            
            for track_idx, det_idx in matches:
                all_matches.append((track_idx, det_idx))
                matched_track_mask[track_idx] = True
                matched_det_mask[det_idx] = True
        
        # TODO Construct new tracks and do initial matchings using first pass matches