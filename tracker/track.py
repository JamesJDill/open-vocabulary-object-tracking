import numpy as np
from enum import IntEnum

class TrackState(IntEnum):
    NEW     = 0
    ACTIVE  = 1
    MISSING = 2
    EXITED = 3
    TERMINATED = 4

X, Y, W, H, VX, VY, VW, VH = range(8)

class Track(object):
    """
    Track object class containing the state information, track status, and visualization information of a given track
    """
    
    __slots__ = (
        "track_id", 
        "state", 
        "mean", 
        "covariance", 
        "frames_missing", 
        "missing_limit",
        "first_frame", 
        "state_history",
        "__weakref__"
    )
    
    def __init__(
        self, 
        track_id: int, 
        mean: np.ndarray, 
        covariance: np.ndarray,
        init_frame: int,
        missing_limit: int=30,
        store_states: bool=False
    ):
        """
        Initializes a new track. 
        mean and covariance should be retrieved from a KalmanFilter.initialize() call

        Args:
            track_id : int
                the unique track identifier assigned to this track
            mean : (8,) ndarray
                Initial state mean [x, y, w, h, vx, vy, vw, vh]
            covariance : (8,8) ndarray
                Initial state covariance
            init_frame : int
                the initial frame of the track, needed for determining track age
            missing_limit : int = 30
                the number of frames a track can be missing before it is removed
            store_states : bool
                whether to store the track states over time
        """
        
        self.track_id = track_id
        self.state = TrackState.NEW
        self.mean = mean
        self.covariance = covariance
        self.frames_missing = 0 # increments if self.state == TrackState.MISSING
        self.missing_limit = missing_limit
        
        self.first_frame = init_frame
        self.state_history = None
        if store_states:
            self.state_history = []
    
    
    def as_dict(self):
        return {
            "id": self.track_id,
            "state": int(self.state),
            "mean": self.mean.tolist(),
            "frames_missing": self.frames_missing,
            "first_frame": self.first_frame,
        }


    def update(
        self,
        new_mean: np.ndarray, 
        new_cov: np.ndarray,
        new_state: TrackState=None,
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
                New track state
        """  
        
        self.mean = new_mean
        self.covariance = new_cov
        
        if new_state is not None:
            self.state = new_state

        if self.state is TrackState.MISSING:
            self.frames_missing += 1
            if self.frames_missing >= self.missing_limit:
                self.state = TrackState.TERMINATED
        elif self.state not in [TrackState.TERMINATED, TrackState.EXITED]:
            self.frames_missing = 0
            
        if self.state_history is not None:
            self.state_history.append(int(self.state))
            
    
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
    def is_active(self):   return self.state is TrackState.ACTIVE
    
    @property
    def is_missing(self):  return self.state is TrackState.MISSING
    
    @property
    def is_dead(self):     return self.state in (TrackState.EXITED, TrackState.TERMINATED)

    @property
    def tlwh(self): return np.array([self.x - self.w/2, self.y - self.h/2, self.w, self.h], dtype=np.float32)
    
    @property
    def tlbr(self): 
        tl = self.tlwh
        return np.array([tl[0], tl[1], tl[0]+tl[2], tl[1]+tl[3]], dtype=np.float32)

    def __repr__(self):
        return (f"Track(id={self.track_id}, state={self.state.name}, "
            f"xywh=({self.x:.1f},{self.y:.1f},{self.w:.1f},{self.h:.1f}), "
            f"miss={self.frames_missing}/{self.missing_limit})")