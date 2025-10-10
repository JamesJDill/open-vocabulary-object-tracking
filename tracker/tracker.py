import numpy as np

from kalman_filter import KalmanFilter
from track import Track, TrackState
from assignment import dist_cost_matrix, app_cost_matrix, linear_assignment

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
    def __init__(self):
        # Shared objects for managing Tracks
        self.kalman_filter = KalmanFilter()
        self.curr_tracks = []
        
        # Adaptive Threshold/Variables
        self.score_thresh = defaultdict(lambda: 0.01) # label --> score_thresh (float)
                
        # Counter variables 
        self.curr_track = 0
        self.curr_frame = 0
    
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
    ):
        """
        the function to be called every frame

        Args:
            boxes : (M,4) np.ndarray
                the current frame bounding boxes (x, y, w, h) 
            scores : (M,L) np.ndarray
                the current frame scores
        """
        N = len(self.curr_tracks)   # Num Tracks
        M = len(boxes)              # Num Dets
        
        # if there are no dets for the current frame we end early
        if M == 0:
            self.curr_frame += 1
            return
        
        L = scores.shape[1]
        
        best_scores = np.max(scores, axis=1)    # (M,)
        labels      = np.argmax(scores, axis=1) # (M,)
        
        # Update the dynamic threshold for all labels based on current detection scores
        self.update_score_thresh(best_scores, labels)
        
        # Process current tracks for matching
        for track in self.curr_tracks:
            track.mean, track.covariance = self.kalman_filter.predict(track.mean, track.covariance) # Forward tracks by 1 frame

        # Compute the gating distances
        gating_distances = []
        for track in self.curr_tracks:
            gating_distances.append(self.kalman_filter.gating_distance(track.mean, track.covariance, boxes))
        gating_distances = np.vstack(gating_distances)  # (N, M) gating distances^2
        
        # Compute the DIoU-based cost
        costs = dist_cost_matrix(self.curr_tracks, boxes)
        
        # Construct masks for cost matrix
        gating_mask = gating_distances > CHI2_INV95[4]
        score_thresholds = np.array([self.score_thresh[l] for l in labels])
        det_mask = best_scores >= score_thresholds
        
        # 1st Pass
        first_pass_costs = costs.copy()
        first_pass_costs[gating_mask] = BIG_NUM
        first_pass_costs[:, ~det_mask] = BIG_NUM
        
        high_matches, unmatched_tracks, unmatched_dets = linear_assignment(first_pass_costs, thresh=0.7)
        
        # 2nd Pass
        if len(high_matches) > 0:
            matched_tracks = high_matches[:, 0]
            matched_dets   = high_matches[:, 1]
            costs[matched_tracks, :] = BIG_NUM
            costs[:, matched_dets]   = BIG_NUM
            
        costs[gating_mask] = BIG_NUM
        
        low_matches, unmatched_tracks, unmatched_dets = linear_assignment(costs, thresh=0.7)
        matches = np.vstack([high_matches, low_matches])