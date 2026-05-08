#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PointStamped
from cv_bridge import CvBridge
from lab2_perception.msg import ObjectCoordinates
import tf2_ros
import tf2_geometry_msgs
from lab2_perception.cfg import PerceptionHSVConfig
from dynamic_reconfigure.server import Server

from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo

# Import YOLO
from ultralytics import YOLO
from enum import Enum


class RobotState(Enum):
    SEARCHING = 0      # Searching for target
    TRACKING = 1       # Unified tracking (curvature control: alignment + approach fusion)
    REACHED = 2        # Reached target


class UnifiedTracker:
    """
    Curvature control + Dynamic blending strategy + Acceleration smoothing
    Replaces the original separate ALIGNING + APPROACHING states
    """
    
    def __init__(self):
        # === Core curvature parameters ===
        self.k_angular = 3.0           # Curvature gain (κ = k_angular * err)
        self.k_linear = 0.5            # Velocity gain
        
        # === Blending strategy parameters ===
        self.d_transition_far = 1.2    # Far distance boundary: pure curvature control
        self.d_transition_near = 0.7   # Near distance boundary: start blending alignment
        self.d_desired = 0.5           # Desired stopping distance
        
        # === Limit parameters ===
        self.kappa_max = 4.0           # Maximum curvature (1/m), corresponding to minimum radius 0.25m
        self.v_max = 0.3               # Maximum linear velocity (m/s)
        self.v_min_effective = 0.08    # Minimum velocity for effective curvature control
        self.v_min_approach = 0.02     # Final approach velocity (to prevent dead zone)
        self.w_max = 0.5               # Maximum angular velocity (rad/s)
        
        # === Acceleration limit parameters (core smoothing mechanism) ===
        self.accel_lim_v = 1.0         # Linear acceleration limit (m/s^2)
        self.decel_lim_v = 2.0         # Linear deceleration limit (higher, allows emergency stop)
        self.accel_lim_w = 2.0         # Angular acceleration limit (rad/s^2)
        self.decel_lim_w = 4.0         # Angular deceleration limit
        
        # === Internal state ===
        self.last_cmd = Twist()        # Last cycle output (for acceleration limiting)
        self.last_time = rospy.Time.now()
        self.err_filt = 0.0            # Error low-pass filter
        self.err_alpha = 0.3           # Error filtering coefficient
        
        # === Convergence detection ===
        self.align_ok_count = 0
        self.align_ok_required = 3     # Requires 3 consecutive frames meeting criteria to determine arrival
        self.align_threshold = 0.08    # Alignment threshold (normalized error)
        self.dist_threshold = 0.06     # Distance threshold (m)
        
    def reset(self):
        """Reset when state changes"""
        self.last_cmd = Twist()
        self.err_filt = 0.0
        self.align_ok_count = 0
        
    def _clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))
    
    def _sign(self, x):
        return 1.0 if x >= 0 else -1.0
    
    def compute(self, err_raw, Z, dt):
        """
        Unified tracking control: curvature control + dynamic blending + acceleration smoothing
        
        Args:
            err_raw: Normalized lateral error [-1, 1]
            Z: Target distance (m)
            dt: Time interval (s)
            
        Returns:
            (Twist, reached_flag): Smoothed velocity command + whether reached
        """
        # ── 1. Error filtering (reduce jitter) ──
        self.err_filt = self.err_alpha * err_raw + (1 - self.err_alpha) * self.err_filt
        err = self.err_filt
        
        distance_error = Z - self.d_desired
        
        # ── 2. Calculate desired velocity v_cmd_raw (distance control) ──
        if distance_error > 0:
            # Still far: proportional approach
            v_cmd_raw = self.k_linear * distance_error
            v_cmd_raw = min(v_cmd_raw, self.v_max)
        else:
            # Too close: slow reverse (asymmetric limit)
            v_cmd_raw = max(self.k_linear * distance_error, -self.v_max * 0.3)
        
        # ── 3. Calculate desired curvature κ (lateral error control) ──
        kappa = self.k_angular * err
        kappa = self._clamp(kappa, -self.kappa_max, self.kappa_max)
        
        # ── 4. Dynamic blending strategy (core! handles v→0 singularity) ──
        
        if Z > self.d_transition_far:
            # ═══════════════════════════════════════
            # Region A: Far distance (Z > 1.2m) → Pure curvature control
            # Feature: v is sufficient, κ is effective, smooth arc approach
            # ═══════════════════════════════════════
            v_cmd = v_cmd_raw
            w_cmd = kappa * v_cmd
            
        elif Z > self.d_transition_near:
            # ═══════════════════════════════════════
            # Region B: Medium distance (0.7m < Z < 1.2m) → Curvature + angle blending
            # Feature: v decreases, κ starts losing effectiveness, gradually introduce direct angle control
            # ═══════════════════════════════════════
            v_cmd = v_cmd_raw
            
            # Blending coefficient: smaller v, larger direct angle control weight
            # blend = 0 (large v) → 1 (small v)
            v_norm = abs(v_cmd) / self.v_min_effective
            blend = self._clamp(1.0 - v_norm, 0.0, 1.0)
            
            # Curvature control component
            w_from_kappa = kappa * v_cmd
            
            # Direct angle control component (fallback, prevents w→0 when v→0)
            w_direct = 2.0 * err  # Direct P control
            w_direct = self._clamp(w_direct, -self.w_max, self.w_max)
            
            # Blend
            w_cmd = (1 - blend) * w_from_kappa + blend * w_direct
            
        else:
            # ═══════════════════════════════════════
            # Region C: Near distance (Z < 0.7m) → Fine alignment + velocity control
            # Feature: Separated control ensures final precision
            # ═══════════════════════════════════════
            
            if abs(err) > self.align_threshold:
                # Not aligned: prioritize rotation, low-speed forward or stop
                v_cmd = self.v_min_approach * (1 if distance_error > 0 else -0.5)
                # But maintain slight forward motion to avoid complete stall
                
                # Direct angle control
                w_cmd = 2.5 * err  # Stronger gain for quick alignment
                w_cmd = self._clamp(w_cmd, -self.w_max, self.w_max)
                
            else:
                # Aligned: pure distance control, fine-tune to point
                v_cmd = v_cmd_raw
                w_cmd = kappa * v_cmd  # Small error still allows curvature control
        
        # ── 5. Pre-limiting (preliminary clipping before acceleration limiting) ──
        v_cmd = self._clamp(v_cmd, -self.v_max * 0.5, self.v_max)
        w_cmd = self._clamp(w_cmd, -self.w_max, self.w_max)
        
        # ── 6. Core: Acceleration smoothing limit (equivalent implementation of VelocitySmoother) ──
        v_smooth, w_smooth = self._apply_acceleration_limit(v_cmd, w_cmd, dt)
        
        # ── 7. Arrival detection ──
        reached = self._check_reached(err, Z)
        
        # Update state
        self.last_cmd.linear.x = v_smooth
        self.last_cmd.angular.z = w_smooth
        
        cmd = Twist()
        cmd.linear.x = v_smooth
        cmd.angular.z = w_smooth
        
        return cmd, reached
    
    def _apply_acceleration_limit(self, v_target, w_target, dt):
        """
        Acceleration limiting: ensures continuous velocity changes to prevent motor shock
        
        Strategy:
        - Calculate desired increments
        - Choose different limits for acceleration/deceleration
        - Apply limits to get smooth output
        """
        v_last = self.last_cmd.linear.x
        w_last = self.last_cmd.angular.z
        
        # Desired increments
        dv = v_target - v_last
        dw = w_target - w_last
        
        # Determine acceleration or deceleration (same direction increment = acceleration, opposite = deceleration)
        # Linear velocity
        if dv * v_last >= 0 or abs(v_last) < 0.01:
            # Same direction or starting from rest → acceleration phase
            max_dv = self.accel_lim_v * dt
        else:
            # Opposite direction → deceleration phase (can be more aggressive)
            max_dv = self.decel_lim_v * dt
        
        # Angular velocity
        if dw * w_last >= 0 or abs(w_last) < 0.01:
            max_dw = self.accel_lim_w * dt
        else:
            max_dw = self.decel_lim_w * dt
        
        # Apply limits
        if abs(dv) > max_dv:
            dv = self._sign(dv) * max_dv
        
        if abs(dw) > max_dw:
            dw = self._sign(dw) * max_dw
        
        v_smooth = v_last + dv
        w_smooth = w_last + dw
        
        return v_smooth, w_smooth
    
    def _check_reached(self, err, Z):
        """Check if target position is reached"""
        dist_err = abs(Z - self.d_desired)
        err_ok = abs(err) < self.align_threshold
        dist_ok = dist_err < self.dist_threshold
        
        if err_ok and dist_ok:
            self.align_ok_count += 1
        else:
            self.align_ok_count = max(0, self.align_ok_count - 1)
        
        return self.align_ok_count >= self.align_ok_required


class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node')
        
        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_colormap = None
               
        # Get detection mode: 'color' (default) or 'yolo'
        self.detect_mode = rospy.get_param('~mode', 'color')
        rospy.loginfo(f"Current perception mode: {self.detect_mode.upper()}")

        # If in YOLO mode, load model
        if self.detect_mode == 'yolo':
            model_path = os.path.expanduser("~/catkin_ws/models/yolo26s.pt")
            rospy.loginfo(f"Loading YOLO model: {model_path} ...")
            self.yolo_model = YOLO(model_path)
            rospy.loginfo("YOLO model loaded successfully!")

        # Use dynamic reconfiguration
        self.hsv = dict()
        self.dr_srv = Server(PerceptionHSVConfig, self.reconfig_cb)
        self.display_scale = rospy.get_param("~display_scale", 0.5)
        
        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        # Publisher
        self.coord_pub = rospy.Publisher('detected_object', ObjectCoordinates, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # ═══════════════════════════════════════
        # New: Unified tracking controller (replaces original separate states)
        # ═══════════════════════════════════════
        self.tracker = UnifiedTracker()
        
        # State machine (simplified: SEARCHING → TRACKING → REACHED)
        self.current_state = RobotState.SEARCHING
        self.target_lost_count = 0
        self.max_lost_frames = 10
        
        # Search parameters
        self.search_angular_vel = 0.3   # Search rotation speed
        self.target_seen_required = 2
        self.target_seen_count = 0
        self.search_lock_duration = 0.5
        self.search_lock_until = None
        
        # Initialize intrinsic parameter variables
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # CameraInfo subscriber
        self.info_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.camera_info_callback)

        # Publish points in odom frame (debugging)
        self.point_odom_pub = rospy.Publisher("/detected_point_odom", PointStamped, queue_size=10)
        
        # Time recording (for dt calculation in acceleration limiting)
        self.last_process_time = rospy.Time.now()
        
        rospy.loginfo("Perception Node Started (Unified Tracking Version)")

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.cx = msg.K[2]
        self.fy = msg.K[4]
        self.cy = msg.K[5]
        self.info_sub.unregister()
        rospy.loginfo(f"Camera Info received: fx={self.fx}, cx={self.cx}")

    def reconfig_cb(self, config, level):
        self.hsv["lower"] = np.array([config.lower_h, config.lower_s, config.lower_v])
        self.hsv["upper"] = np.array([config.upper_h, config.upper_s, config.upper_v])
        self.display_scale = config.display_scale
        return config

    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image)
        except Exception as e:
            rospy.logerr(f"RGB error: {e}")

    def depth_callback(self, msg):
        try:
            rospy.loginfo_once("[Success] Successfully connected to the depth image topic!")
            depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            self.latest_depth = depth_img
            
            if self.latest_depth is not None:
                depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        except Exception as e:
            rospy.logerr(f"[Depth Callback Error]:{e}")

    def process_image(self, rgb_image):
        # Initialization
        target_found = False
        cX, cY = None, None
        X, Y, Z = None, None, None
        cmd = Twist()
        
        hsv_result = None
        edges = None

        # ═══════════════════════════════════════
        # Part 1: Target detection (obtain cX, cY)
        # ═══════════════════════════════════════
        
        if self.detect_mode == 'color':
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, self.hsv["lower"], self.hsv["upper"])        
            hsv_result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
            
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_image, contours_canny, -1, (0, 255, 0), 3) 
            
            contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_mask:
                c = max(contours_mask, key=cv2.contourArea)
                if cv2.contourArea(c) > 100:
                    target_found = True
                    cv2.drawContours(rgb_image, [c], -1, (0, 0, 255), 3)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

        elif self.detect_mode == 'yolo':
            results = self.yolo_model(rgb_image, conf=0.5, verbose=False, device='cpu')
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                target_found = True
                
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                cv2.rectangle(rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(rgb_image, f"{cls_name}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ═══════════════════════════════════════
        # Part 2: State machine + unified tracking control
        # ═══════════════════════════════════════
        
        # Calculate dt (for acceleration limiting)
        now = rospy.Time.now()
        dt = (now - self.last_process_time).to_sec()
        self.last_process_time = now
        dt = max(dt, 0.001)  # Prevent division by zero
        
        if target_found and cX is not None and cY is not None:
            self.target_lost_count = 0
            cv2.circle(rgb_image, (cX, cY), 7, (255, 255, 255), -1)
            
            if self.latest_depth is not None:
                h, w = self.latest_depth.shape
                if 0 <= cX < w and 0 <= cY < h:
                    # Calculate 3D coordinates
                    X, Y, Z = self.calculate_3d_coordinates(cX, cY, self.latest_depth)
                    self.publish_point_in_odom(X, Y, Z)
                    
                    # Publish coordinate message
                    msg = ObjectCoordinates()
                    msg.x = X
                    msg.y = Y
                    msg.z = Z
                    self.coord_pub.publish(msg)
                    
                    # Calculate normalized lateral error
                    image_center_x = w / 2.0
                    horizontal_error = (cX - image_center_x) / image_center_x
                    
                    # ── State machine processing ──
                    if self.current_state == RobotState.SEARCHING:
                        self.target_seen_count += 1
                        if self.target_seen_count == 1:
                            self.search_lock_until = rospy.Time.now() + rospy.Duration(self.search_lock_duration)
                        
                        if self.target_seen_count >= self.target_seen_required:
                            # Switch to tracking
                            self.current_state = RobotState.TRACKING
                            self.tracker.reset()  # Reset controller state
                            rospy.loginfo("State: SEARCHING → TRACKING")
                        else:
                            # Pause during lock period
                            if self.search_lock_until and rospy.Time.now() < self.search_lock_until:
                                cmd.angular.z = 0.0
                                cmd.linear.x = 0.0
                    
                    elif self.current_state == RobotState.TRACKING:
                        # ═══════════════════════════════════════
                        # Core: Unified tracking control (curvature + blending + smoothing)
                        # ═══════════════════════════════════════
                        cmd, reached = self.tracker.compute(horizontal_error, Z, dt)
                        
                        if reached:
                            self.current_state = RobotState.REACHED
                            rospy.loginfo("State: TRACKING → REACHED")
                    
                    elif self.current_state == RobotState.REACHED:
                        # Maintain at target position (fine-tuning and holding)
                        cmd, _ = self.tracker.compute(horizontal_error, Z, dt)
                        
                        # If deviated too far, restart tracking
                        if abs(horizontal_error) > self.tracker.align_threshold * 2 or \
                           abs(Z - self.d_desired) > self.tracker.dist_threshold * 3:
                            self.current_state = RobotState.TRACKING
                            self.tracker.reset()
                            rospy.loginfo("State: REACHED → TRACKING (drift detected)")
                    
                    # Draw debugging information
                    self._draw_debug_info(rgb_image, cX, cY, w, h, 
                                          horizontal_error, Z, cmd)
                    
        # ── Target loss handling ──
        if not target_found:
            self.target_lost_count += 1
            self.target_seen_count = 0
            
            if self.target_lost_count > self.max_lost_frames:
                if self.current_state != RobotState.SEARCHING:
                    self.current_state = RobotState.SEARCHING
                    self.tracker.reset()
                    rospy.loginfo("State: → SEARCHING (target lost)")
                
                # Search rotation
                cmd.angular.z = self.search_angular_vel
                cmd.linear.x = 0.0

        # Diagnostic printing
        self._print_diagnostics(target_found, cX, cY, X, Y, Z, cmd)

        # Publish control command
        self.cmd_vel_pub.publish(cmd)

        # Display images
        cv2.imshow("Detection Result (RGB)", self.resize(rgb_image))
        if self.detect_mode == 'color' and hsv_result is not None and edges is not None:
            cv2.imshow("HSV Result", self.resize(hsv_result))
            cv2.imshow("Canny Edges", self.resize(edges))
        if self.depth_colormap is not None:
            cv2.imshow("Depth Colormap", self.resize(self.depth_colormap))
        
        cv2.waitKey(1)

    def _draw_debug_info(self, rgb_image, cX, cY, w, h, 
                         horizontal_error, Z, cmd):
        """Draw debugging information on image"""
        # State
        state_text = f"State: {self.current_state.name}"
        cv2.putText(rgb_image, state_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Coordinates
        coord_text = f"Z:{Z:.2f}m err:{horizontal_error:.3f}"
        cv2.putText(rgb_image, coord_text, (cX - 50, cY - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Velocity command
        vel_text = f"v:{cmd.linear.x:.3f} w:{cmd.angular.z:.3f}"
        cv2.putText(rgb_image, vel_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Curvature (if v is not zero)
        if abs(cmd.linear.x) > 0.01:
            kappa = cmd.angular.z / cmd.linear.x
            kappa_text = f"k:{kappa:.2f}"
            cv2.putText(rgb_image, kappa_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Center crosshair
        cx_int = int(w / 2)
        cy_int = int(h / 2)
        cv2.line(rgb_image, (cx_int - 20, cy_int), (cx_int + 20, cy_int), (0, 255, 255), 2)
        cv2.line(rgb_image, (cx_int, cy_int - 20), (cx_int, cy_int + 20), (0, 255, 255), 2)

    def _print_diagnostics(self, target_found, cX, cY, X, Y, Z, cmd):
        """Print diagnostic information"""
        if not target_found:
            rospy.logwarn(f"[Perception] Target not found ({self.detect_mode})")
        else:
            rospy.loginfo(f"[Perception] Target locked! cX={cX}, cY={cY}")
            if self.latest_depth is None:
                rospy.logerr("[Perception] Depth image missing!")
            elif X is not None:
                rospy.loginfo(f"[Perception] 3D: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
            else:
                rospy.logwarn("[Perception] Invalid 3D coords")
        
        rospy.loginfo(f"[State] {self.current_state.name}")
        rospy.loginfo(f"[Cmd] v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}")
        rospy.loginfo("-" * 55)

    def resize(self, image):
        self.display_scale = rospy.get_param("~display_scale", 0.5)
        return cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, 
                         interpolation=cv2.INTER_AREA)

    def calculate_3d_coordinates(self, u, v, depth_image):
        """3D coordinate calculation based on ROI median filtering"""
        if self.fx is None:
            fx, fy, cx, cy = 554.25, 554.25, 320.5, 240.5
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        h, w = depth_image.shape
        box_size = 10
        u_min = max(0, u - box_size)
        u_max = min(w - 1, u + box_size)
        v_min = max(0, v - box_size)
        v_max = min(h - 1, v + box_size)

        roi = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = roi[(roi > 0.0) & (~np.isnan(roi))]

        if len(valid_depths) > 0:
            Z = float(np.median(valid_depths))
            if Z > 10.0:
                Z = Z / 1000.0
        else:
            return 0.0, 0.0, 0.0

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return float(X), float(Y), float(Z)

    def publish_point_in_odom(self, X, Y, Z):
        point_cam = PointStamped()
        point_cam.header.frame_id = "camera_rgb_optical_frame"
        point_cam.header.stamp = rospy.Time(0)
        point_cam.point.x = float(X)
        point_cam.point.y = float(Y)
        point_cam.point.z = float(Z)

        try:
            if not self.tf_buffer.can_transform("odom", point_cam.header.frame_id, 
                                               rospy.Time(0), rospy.Duration(0.2)):
                return
            point_odom = self.tf_buffer.transform(point_cam, "odom", rospy.Duration(0.2))
            self.point_odom_pub.publish(point_odom)
        except Exception as e:
            pass

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = PerceptionNode()
    node.run()