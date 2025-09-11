import numpy as np

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    area1 = max(0, (x1_max - x1_min)) * max(0, (y1_max - y1_min))
    area2 = max(0, (x2_max - x2_min)) * max(0, (y2_max - y2_min))
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def iou_vectorized(tracks: np.ndarray, detections: np.ndarray):
    """
    Vectorized IoU between tracks and detections
    
    tracks: (N,4) array of xyxy boxes
    detections: (M,4) array of xyxy boxes
    returns: (N,M) IoU matrix
    """
    N, M = tracks.shape[0], detections.shape[0]
        
    detections = detections.T
    d_x1, d_y1, d_x2, d_y2 = np.vsplit(detections, 4)   # (1, M)
    t_x1, t_y1, t_x2, t_y2 = np.hsplit(tracks, 4)       # (N, 1)
    
    d_area = (d_x2 - d_x1) * (d_y2 - d_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    
    i_x1, i_y1 = np.maximum(t_x1, d_x1), np.maximum(t_y1, d_y1)
    i_x2, i_y2 = np.minimum(t_x2, d_x2), np.minimum(t_y2, d_y2)
    
    inter_area = np.maximum(i_y2 - i_y1, 0) * np.maximum(i_x2 - i_x1, 0)
    union_area = t_area + d_area - inter_area

    return np.divide(inter_area, union_area, np.zeros((N,M)), where=union_area > 0)

def cdist_vectorized(tracks: np.ndarray, detections: np.ndarray):
    """
    Vectorized CDist variable computation
    
    tracks: (N,4) array of xyxy boxes
    detections: (M,4) array of xyxy boxes
    returns: (N,M) CDist matrix
    """
    detections = detections.T
    d_x1, d_y1, d_x2, d_y2 = np.vsplit(detections, 4)   # (1, M)
    t_x1, t_y1, t_x2, t_y2 = np.hsplit(tracks, 4)       # (N, 1)
    
    d_cx, d_cy = (d_x1 + d_x2) / 2, (d_y1 + d_y2) / 2
    t_cx, t_cy = (t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2
    
    c2 = (d_cx - t_cx)**2 + (d_cy - t_cy)**2
    
    x_max, y_max = np.maximum(d_x2, t_x2), np.maximum(d_y2, t_y2)
    x_min, y_min = np.minimum(d_x1, t_x1), np.minimum(d_y1, t_y1)
    
    span2 = (x_max - x_min)**2 + (y_max - y_min)**2
    
    if span2 == 0:
        return 0
    
    return c2 / span2
    
def cost_matrix(tracks: np.ndarray, detections: np.ndarray, alpha=0.7, gamma=1.5):
    """
    Computes the DIoU-based cost matrix between Tracks and detections
    
    returns: (N,M) Cost matrix
    """
    N, M = tracks.shape[0], detections.shape[0]
    ones = np.ones((N,M))
    
    iou = iou_vectorized(tracks, detections)
    cdist = cdist_vectorized(tracks, detections)
    w = (1.0 - iou)**gamma
    
    return (ones - iou) + (alpha * w * cdist)
