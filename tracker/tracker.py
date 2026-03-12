import numpy as np

from .kalman_filter import KalmanFilter
from .track import Track, TrackState
from .assignment import dist_cost_matrix, app_cost_matrix, linear_assignment, iou_one_to_many

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
            
            new_thresh = max(0.01, np.mean(filt_scores) - 0.01)
            if len(filt_scores) == 1:
                new_thresh = max(0.01, filt_scores[0] - 0.01)
            
            if self.score_thresh[label] == 0.01:
                self.score_thresh[label] = new_thresh
            
            self.score_thresh[label] = 0.8 * self.score_thresh[label] + 0.2 * new_thresh
    
    def _xywh_to_xyxy(self, boxes_xywh: np.ndarray) -> np.ndarray:
        # boxes are center-based [x, y, w, h]
        x, y, w, h = boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,2], boxes_xywh[:,3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x1 + w
        y2 = y1 + h
        return np.stack([x1, y1, x2, y2], axis=1)

    def _classwise_nms(
            self,
            boxes_xywh: np.ndarray,
            scores: np.ndarray,
            labels: np.ndarray,
            best_scores: np.ndarray,
            iou_thresh: float = 0.5,
        ):
        """
        Apply NMS only among detections that share the same argmax label.

        Args:
            boxes_xywh: (M,4) center-based boxes
            scores: (M,L) full score vectors
            labels: (M,) argmax label per detection
            best_scores: (M,) best score per detection
            iou_thresh: suppress same-label boxes with IoU > this threshold

        Returns:
            filtered_boxes_xywh, filtered_scores, filtered_labels, filtered_best_scores
        """
        if len(boxes_xywh) == 0:
            return boxes_xywh, scores, labels, best_scores

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        keep_indices = []

        for label in np.unique(labels):
            cls_idx = np.where(labels == label)[0]
            cls_boxes = boxes_xyxy[cls_idx]
            cls_best = best_scores[cls_idx]

            order = np.argsort(-cls_best)
            cls_keep = []

            while len(order) > 0:
                i = order[0]
                cls_keep.append(cls_idx[i])

                if len(order) == 1:
                    break

                ious = iou_one_to_many(cls_boxes[i], cls_boxes[order[1:]])
                remaining = np.where(ious <= iou_thresh)[0]
                order = order[remaining + 1]

            keep_indices.extend(cls_keep)

        keep_indices = np.array(sorted(keep_indices), dtype=int)
        
        return (
            boxes_xywh[keep_indices],
            scores[keep_indices],
            labels[keep_indices],
            best_scores[keep_indices],
        )
    
    def extract_track_boxes(self):
        if len(self.curr_tracks) == 0:
            return np.empty((0,4), dtype=np.float64)
        arr = np.array([[t.x, t.y, t.w, t.h] for t in self.curr_tracks], dtype=np.float64)
        return self._xywh_to_xyxy(arr)
    
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
        self.curr_frame += 1
        
        best_scores = np.max(scores, axis=1)    # (M,)
        labels      = np.argmax(scores, axis=1) # (M,)
        
        # Apply NMS only within the same predicted label
        boxes, scores, labels, best_scores = self._classwise_nms(
            boxes_xywh=boxes,
            scores=scores,
            labels=labels,
            best_scores=best_scores,
            iou_thresh=0.5,
        )
        
        N = len(self.curr_tracks)   # Num Tracks
        M = len(boxes)              # Num Dets
        
        # If we have no current dets to spawn as tracks
        if M == 0:
            # If we have current tracks to update
            if N > 0:
                for track in self.curr_tracks:
                    new_mean, new_covariance = self.kalman_filter.predict(track.mean, track.covariance)
                    track.frames_missing += 1    
                    new_state = TrackState.TERMINATED if track.frames_missing >= track.missing_limit else TrackState.MISSING
                    track.update(new_mean, new_covariance, new_state, new_scores=None)
                
                # Purge Terminated Tracks
                self.curr_tracks = [t for t in self.curr_tracks if t.state != TrackState.TERMINATED]  
            return
        
        # Update the dynamic threshold for all labels based on current detection scores
        self.update_score_thresh(best_scores, labels)
        
        # If we have no current tracks to update
        if N == 0:
            # If we have dets we can spawn them as tracks
            if M > 0:
                for i in range(M):
                    if best_scores[i] >= self.score_thresh[labels[i]]:
                        init_scores = scores[i]
                        init_mean, init_cov = self.kalman_filter.initialize(boxes[i])
                        
                        new_track = Track(
                            self.curr_track, 
                            labels[i], 
                            init_scores, 
                            init_mean, 
                            init_cov
                        )
                        self.curr_tracks.append(new_track)
                        self.curr_track += 1
            return
        
        # Process current tracks for matching
        for track in self.curr_tracks:
            track.mean, track.covariance = self.kalman_filter.predict(track.mean, track.covariance) # Forward tracks by 1 frame

        # Compute the gating distances
        gating_distances = []
        for track in self.curr_tracks:
            gating_distances.append(
                self.kalman_filter.gating_distance(track.mean, track.covariance, boxes)
                )
        gating_distances = np.vstack(gating_distances)  # (N, M) gating distances^2
        
        # Compute the DIoU-based cost
        track_boxes = self.extract_track_boxes()
        det_boxes = self._xywh_to_xyxy(boxes)
        dist_costs = dist_cost_matrix(track_boxes, det_boxes)
        
        # Compute Appearance-based cost
        track_score_vectors = np.stack([t.score_vector for t in self.curr_tracks], axis=0)
        score_costs = app_cost_matrix(track_score_vectors, scores)
        
        costs = 0.8 * dist_costs + 0.2 * score_costs
        
        # Construct masks for cost matrix
        gating_mask = gating_distances > CHI2_INV95[4]
        score_thresholds = np.array([self.score_thresh[l] for l in labels])
        det_mask = best_scores >= score_thresholds
        
        # ---------- First pass ----------
        first_pass_costs = costs.copy()
        first_pass_costs[gating_mask] = BIG_NUM
        first_pass_costs[:, ~det_mask] = BIG_NUM
        
        high_matches, unmatched_tracks, unmatched_dets = linear_assignment(first_pass_costs, thresh=0.7)
        
        # ---------- Second pass ----------
        low_matches = np.empty((0, 2), dtype=int)
        if len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            second_costs = costs[np.ix_(unmatched_tracks, unmatched_dets)].copy()
            second_gating = gating_mask[np.ix_(unmatched_tracks, unmatched_dets)]
            second_costs[second_gating] = BIG_NUM

            low_local_matches, second_unmatched_tracks, second_unmatched_dets = linear_assignment(
                second_costs, thresh=0.7
            )

            if len(low_local_matches) > 0:
                low_matches = np.column_stack([
                    unmatched_tracks[low_local_matches[:, 0]],
                    unmatched_dets[low_local_matches[:, 1]],
                ])

            unmatched_tracks = unmatched_tracks[second_unmatched_tracks]
            unmatched_dets = unmatched_dets[second_unmatched_dets]

        matches = np.vstack([high_matches, low_matches]) if len(low_matches) > 0 else high_matches
        
        # ---------- Post matching assignment ----------
        
        # 1. Update Matched Tracks
        for track_idx, det_idx in matches:
            track = self.curr_tracks[track_idx]
            new_mean, new_cov = self.kalman_filter.update(track.mean, track.covariance, boxes[det_idx])
            track.frames_missing = 0
            track.update(new_mean, new_cov, new_state=TrackState.ACTIVE, new_scores=scores[det_idx])
            
        # 2. Update Unmatched Tracks
        for track_idx in unmatched_tracks:
            track = self.curr_tracks[track_idx]
            track.frames_missing += 1
            new_state = TrackState.TERMINATED if track.frames_missing >= track.missing_limit else TrackState.MISSING
            track.update(track.mean, track.covariance, new_state, new_scores=None)
            
        # 3. Spawn new tracks from unmatched detections which pass det threshold
        for det_idx in unmatched_dets:
            if best_scores[det_idx] >= self.score_thresh[labels[det_idx]]:
                init_mean, init_cov = self.kalman_filter.initialize(boxes[det_idx])
                new_track = Track(self.curr_track, labels[det_idx], scores[det_idx], init_mean, init_cov)
                
                self.curr_tracks.append(new_track)
                self.curr_track += 1
                
        # 4. Purge Terminated Tracks
        self.curr_tracks = [t for t in self.curr_tracks if t.state != TrackState.TERMINATED]