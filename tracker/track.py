import numpy as np
from enum import IntEnum

class TrackState(IntEnum):
    ACTIVE  = 1
    MISSING = 2
    TERMINATED = 3

X, Y, W, H, VX, VY, VW, VH = range(8)

class Track(object):
    """
    Track object class containing the state information, track status, and visualization information of a given track
    """
    
    __slots__ = (
        "track_id", 
        "label",
        "app_vector",
        "state", 
        "mean",
        "covariance", 
        "frames_missing",
        "missing_limit",
        "__weakref__"
    )
    
    def __init__(
        self, 
        track_id: int, 
        label: int,
        init_scores: np.ndarray,
        mean: np.ndarray, 
        covariance: np.ndarray,
        missing_limit: int=30
    ):
        """
        Initializes a new track. 
        mean and covariance should be retrieved from a KalmanFilter.initialize() call

        Args:
            track_id : int
                the unique track identifier assigned to this track
            label : int
                the label of the track
            init_scores : (L) ndarray
                initial scores for each label of the current track
            mean : (8,) ndarray
                Initial state mean [x, y, w, h, vx, vy, vw, vh]
            covariance : (8,8) ndarray
                Initial state covariance
            missing_limit : int = 30
                the number of frames a track can be missing before it is removed
        """
        self.track_id = track_id
        self.label = label
        self.app_vector = init_scores
        self.state = TrackState.ACTIVE
        self.mean = mean
        self.covariance = covariance
        self.frames_missing = 0 # increments if self.state == TrackState.MISSING
        self.missing_limit = missing_limit

    def update_app_vector(
        self, 
        new_scores: np.ndarray
    ):
        """
        EMA update of the appearance vector

        Args:
            new_scores (L,): ndarray, latest per-label scores for this track
        """
        alpha = 2.0 / (self.missing_limit + 1.0)
        self.app_vector = (1.0 - alpha) * self.app_vector + alpha * new_scores.flatten()


    def update(
        self,
        new_mean: np.ndarray, 
        new_cov: np.ndarray,
        new_state: TrackState,
        new_scores: np.ndarray = None,
    ):
        """
        Updates the track with a new mean, covariance, and state. 
        This should be called on each track after a single pass of a kalman filter.
        
        Args:
            new_mean : (8,) ndarray
                new state mean [x, y, w, h, vx, vy, vw, vh]
            new_cov : (8,8) ndarray
                new state covariance
            new_state : TrackState
                new track state for the track
            new_scores: (L,) ndarray
                new scores for the matched track
        """  
        
        self.mean = new_mean
        self.covariance = new_cov
        self.state = new_state
        
        if new_scores is not None:
            self.update_app_vector(new_scores)
            
    
    # For extracting values from the track state mean
    # Usually for visualization or debugging
    @property
    def x(self):  return float(self.mean[X])
    
    @property
    def y(self):  return float(self.mean[Y])
    
    @property
    def w(self):  return float(self.mean[W])
    
    @property
    def h(self):  return float(self.mean[H])
    
    @property
    def vx(self): return float(self.mean[VX])
    
    @property
    def vy(self): return float(self.mean[VY])
    
    @property
    def vw(self): return float(self.mean[VW])
    
    @property
    def vh(self): return float(self.mean[VH])
    
    @property
    def center(self): return self.mean[[X, Y]].copy()
    
    @property
    def area(self):   return float(self.w * self.h)
    
    @property
    def score_vector(self): return self.app_vector
    
    @property
    def tlwh(self): return np.array([self.x - self.w/2, self.y - self.h/2, self.w, self.h], dtype=np.float32)
    
    @property
    def tlbr(self): 
        tl = self.tlwh
        return np.array([tl[0], tl[1], tl[0]+tl[2], tl[1]+tl[3]], dtype=np.float32)