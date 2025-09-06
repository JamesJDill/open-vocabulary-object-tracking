import numpy as np

class KalmanFilter(object):
    """
    A Kalman filter for tracking bounding boxes
    
    8-dimensional state space vector
        x, y, w, h, vx, vy, vw, vh
    containing the bounding box center position (x, y), width w, height h, and their linear velocities.
    
    This implementation uses Mahalanobis distance for gating incorrect detections
    
    The following class will use the following definitions (im following equations I've written down)
        x = mean vector / state space vector for the current detection
        z = measurement vector of a bounding box
        F = forward matrix / dynamics space matrix
        H = measurement space matrix
        P = covariance matrix
        R = measurement noise matrix
        Q = process noise covariance
        K = kalman gain
        S = innovation covariance matrix / system uncertainty
    """
    
    def __init__(self, dt=1):
        self.dt = dt
        I4 = np.eye(4)
        Z4 = np.zeros((4, 4))
        self.F = np.block([[I4, dt * I4], [Z4, I4]])    # dynamics space matrix (forward matrix)
        self.H = np.block([I4, Z4])                     # measurement space matrix

        # BoT-SORT implementations, supposedly tested
        # Adaptive values / methods for noise will be done later
        self._std_pos = 1. / 20
        self._std_vel = 1. / 160


    def initialize(self, z):
        """
        Create a new track from an unassociated measurement z = [x, y, w, h].
        Returns initial state x0 and covariance P0.

        Args
            z : (4,) ndarray
                bounding box coordinates [x, y, w, h] with center positions (x, y) and width w, height h
        
        Returns
            x : (8,) ndarray
                Initial state mean [x, y, w, h, vx, vy, vw, vh]
            P : (8,8) ndarray
                Initial state covariance
            
        Initial velocities mean vector should be set to 0.
        """
        
        std = [
            2 * self._std_pos * z[2],
            2 * self._std_pos * z[3],
            2 * self._std_pos * z[2],
            2 * self._std_pos * z[3],
            10 * self._std_vel * z[2],
            10 * self._std_vel * z[3],
            10 * self._std_vel * z[2],
            10 * self._std_vel * z[3]
        ]
        P = np.diag(np.square(std))
        
        zeros = np.zeros_like(z)
        x = np.r_[z, zeros]
        
        return x, P
    
    
    def predict(self, x, P):
        """
        Runs the Kalman Filter prediction step
        
        Args
            x : (8,) ndarray
                State mean at the previous frame: [x, y, w, h, vx, vy, vw, vh]
            P : (8,8) ndarray
                State covariance at the previous frame
            
        Returns
            x_pred : (8,) ndarray
                Predicted state mean after advancing one frame by the motion model F: x_pred = F @ x
            P_pred : (8,8) ndarray
                Predicted covariance propagated through the dynamics and inflated by: P_pred = F @ P @ F.T + Q
        """
        
        std = [
            self._std_pos * x[2],
            self._std_pos * x[3],
            self._std_pos * x[2],
            self._std_pos * x[3],
            self._std_vel * x[2],
            self._std_vel * x[3],
            self._std_vel * x[2],
            self._std_vel * x[3]
        ]
        Q = np.diag(np.square(std)) # Process noise covariance matrix
        
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + Q
        
        return x_pred, P_pred
    
    
    def project(self, x_pred, P_pred):
        """
        Projects the predicted state mean and covariance into measurement space.
        
        Args
            x_pred : (8,) ndarray
                Predicted state mean after advancing one frame by the motion model F: x_pred = F @ x
            P_pred : (8,8) ndarray
                Predicted covariance propagated through the dynamics and inflated by: P_pred = F @ P @ F.T + Q
        
        Returns
            z_pred : (4,) ndarray
                Predicted measurement bounding box after advancing by one frame: z_pred = H @ x_pred
            S : (4,4) ndarray
                Innovation covariance matrix: H @ P_pred @ H.T + R
        """
        
        std = [
            self._std_pos * x_pred[2],
            self._std_pos * x_pred[3],
            self._std_pos * x_pred[2],
            self._std_pos * x_pred[3]
        ]
        R = np.diag(np.square(std)) # Measurement noise covariance matrix
        
        S = self.H @ P_pred @ self.H.T + R
        z_pred = self.H @ x_pred
        
        return z_pred, S
        
        
    def update(self, x_pred, P_pred, z):
        """
        Runs the Kalman Filter update step.

        Args
            x_pred : (8,) ndarray
                Predicted state mean after advancing one frame by the motion model F: x_pred = F @ x
            P_pred : (8,8) ndarray
                Predicted covariance propagated through the dynamics and inflated by: P_pred = F @ P @ F.T + Q
            z : (4,) ndarray
                bounding box coordinates [x, y, w, h] with center positions (x, y) and width w, height h
        
        Returns
            x_new : (8,) ndarray
                the new state mean after processing the predicted state mean and Kalman gain
            P_new : (8,8) ndarray
                the new covariance matrix after processing the predicted covariance matrix and innovation covariance matrix
        """
        
        z_pred, S = self.project(x_pred, P_pred)
        y = z - z_pred # innovation
        
        PHt = P_pred @ self.H.T          # (n,m)
        L   = np.linalg.cholesky(S)      # (m,m)
        U   = np.linalg.solve(L, PHt.T)  # (m,n)  solves L U = PHt^T
        KT  = np.linalg.solve(L.T, U)    # (m,n)  solves L^T K^T = U
        K   = KT.T                       # (n,m)
        
        x_new = x_pred + K @ y
        P_new = P_pred - K @ S @ K.T
        
        return x_new, P_new
        

    def gating_distance(self, x_pred, P_pred, z, only_position=False):
        """
        Compute gating distance between state distribution and cuurrent frame bounding boxes.
        If only_position is true then use dof=2, else dof=4.

        Args
            x_pred : (8,) ndarray
                Predicted state mean after advancing one frame by the motion model F: x_pred = F @ x
            P_pred : (8,8) ndarray
                Predicted covariance propagated through the dynamics and inflated by: P_pred = F @ P @ F.T + Q
            z : (N,4) ndarray
                bounding box coordinates [x, y, w, h] with center positions (x, y) and width w, height h
            only_position
                Determines whether to use 2 dof or 4 dof for chi square distribution. Defaults to False.
        
        Returns
            distances : (N,) squared mahalanobis distances between (x_pred, P_pred) and measurements z
        """
        z_pred, S = self.project(x_pred, P_pred)
        
        if only_position:
            z_pred, S = z_pred[:2], S[:2, :2]
            z = z[:, :2]
        
        d = z - z_pred               # (N, d)
        L = np.linalg.cholesky(S)    # d×d
        y = np.linalg.solve(L, d.T)  # d×N  (whitened residuals)
        d2 = np.sum(y*y, axis=0)     # length-N, squared Mahalanobis
        return d2