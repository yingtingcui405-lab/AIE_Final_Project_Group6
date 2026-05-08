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

# 导入 YOLO
from ultralytics import YOLO
from enum import Enum

class RobotState(Enum):
    SEARCHING = 0      # 搜索目标
    ALIGNING = 1       # 对准目标(旋转调整)
    APPROACHING = 2    # 接近目标(前进)
    REACHED = 3        # 到达目标


class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node')
        
        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_colormap = None
               
        # 获取检测模式：'color' (默认) 或 'yolo'
        self.detect_mode = rospy.get_param('~mode', 'color')
        rospy.loginfo(f"Current perception mode: {self.detect_mode.upper()}")

        # 如果是 YOLO 模式，则加载模型
        if self.detect_mode == 'yolo':
            model_path = os.path.expanduser("~/catkin_ws/models/yolo26s.pt")
            rospy.loginfo(f"Loading YOLO model: {model_path} ...")
            self.yolo_model = YOLO(model_path)
            rospy.loginfo("YOLO model loaded successfully!")

        # 使用动态重新配置
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

        # 状态机相关
        self.current_state = RobotState.SEARCHING
        self.target_lost_count = 0
        self.max_lost_frames = 10  # 连续丢失多少帧后重新搜索
        self.target_seen_count = 0
        self.target_seen_required = rospy.get_param("~target_seen_required", 2)
        self.search_lock_until = None
        self.search_lock_duration = rospy.get_param("~search_lock_duration", 0.5)
        
        # 控制参数
        self.align_threshold = 0.15  # 对准阈值(归一化坐标)
        self.distance_threshold = 0.1  # 距离到达阈值(米)
        self.desired_distance = 0.5  # 期望距离(米)
        
        # PID控制参数
        self.angular_kp = 0.5
        self.linear_kp = 0.5
        self.max_angular_speed = 0.3
        self.max_linear_speed = 0.2

        # 误差滤波与稳定控制参数
        self.error_alpha = 1.0       
        self.horizontal_error_filt = 0.0
        self.angular_deadband = 0.03
        self.min_angular_speed = 0.03
        self.prev_horizontal_error = None
        # 分段控制阈值
        self.angular_fast_threshold = 0.12
        self.angular_slow_threshold = 0.05

        # 对准稳定计数
        self.align_ok_count = 0
        self.align_bad_count = 0
        self.align_ok_required = 1   
        self.align_bad_required = 3

        # 距离误差滤波与限幅
        self.dist_alpha = 1.0        
        self.max_distance_error = 1.0

        # 初始化内参变量
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # 添加 CameraInfo 订阅者
        self.info_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.camera_info_callback)

        # 发布 odom 下的点（可选：用于调试查看）
        self.point_odom_pub = rospy.Publisher("/detected_point_odom", PointStamped, queue_size=10)

        # marker id 计数
        self.marker_id = 0
        
        rospy.loginfo("Perception Node Started")

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
            # 1. 打印一句话，证明程序确实收到了话题数据 (只会打印一次)
            rospy.loginfo_once("[Success] Successfully connected to the depth image topic!")
            
            # 2. 改用 "passthrough" 自动适配格式，避免 16UC1 和 32FC1 冲突
            depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            
            # 3. 确保 numpy 存在
            import numpy as np 
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.latest_depth = depth_img
            
            # 4. 确保 cv2 存在
            if self.latest_depth is not None:
                import cv2 
                depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
        except Exception as e:
            # 5. 如果出错，打印出极其醒目的红色报错
            rospy.logerr(f"[Depth Callback Error]:{e}")

    def process_image(self, rgb_image):
        # 初始化变量，防止报错
        target_found = False
        cX, cY = None, None
        X, Y, Z = None, None, None
        cmd = Twist()
        
        hsv_result = None
        edges = None

        # ==================== 第 1 部分：获取二维坐标 cX, cY ====================
        
        # 模式 1：基于颜色的检测
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

        # 模式 2：基于 YOLO 的检测
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
                # rospy.loginfo_throttle(1.0, f"YOLO 锁定了目标: 【{cls_name}】")

        # ==================== 第 2 部分：3D坐标计算与状态机 ====================
        
        if target_found and cX is not None and cY is not None:
            self.target_lost_count = 0
            cv2.circle(rgb_image, (cX, cY), 7, (255, 255, 255), -1)
            
            if self.latest_depth is not None:
                h, w = self.latest_depth.shape
                if 0 <= cX < w and 0 <= cY < h:
                    X, Y, Z = self.calculate_3d_coordinates(cX, cY, self.latest_depth)
                    self.publish_point_in_odom(X, Y, Z)
                    
                    msg = ObjectCoordinates()
                    msg.x = X
                    msg.y = Y
                    msg.z = Z
                    self.coord_pub.publish(msg)
                    
                    image_center_x = w / 2.0
                    horizontal_error_raw = (cX - image_center_x) / image_center_x
                    self.horizontal_error_filt = (
                        self.error_alpha * horizontal_error_raw
                        + (1.0 - self.error_alpha) * self.horizontal_error_filt
                    )
                    horizontal_error = self.horizontal_error_filt
                    current_distance = Z
                    depth_valid = (not np.isnan(current_distance)) and (current_distance > 0.0)
                    if depth_valid:
                        distance_error = current_distance - self.desired_distance
                    else:
                        distance_error = 0.0
                    
                    if self.current_state == RobotState.SEARCHING:
                        self.target_seen_count += 1
                        if self.target_seen_count == 1:
                            self.search_lock_until = rospy.Time.now() + rospy.Duration(self.search_lock_duration)
                        if self.target_seen_count >= self.target_seen_required:
                            self.current_state = RobotState.ALIGNING
                        else:
                            if self.search_lock_until is not None and rospy.Time.now() < self.search_lock_until:
                                cmd.angular.z = 0.0
                                cmd.linear.x = 0.0
                    
                    elif self.current_state == RobotState.ALIGNING:
                        if abs(horizontal_error) > self.align_threshold:
                            self.align_ok_count = 0
                            self.align_bad_count += 1
                            cmd.angular.z = self.compute_angular_cmd(horizontal_error)
                            cmd.linear.x = 0.0
                        else:
                            self.align_bad_count = 0
                            self.align_ok_count += 1
                            if self.align_ok_count >= self.align_ok_required:
                                self.current_state = RobotState.APPROACHING
                                self.align_ok_count = 0
                    
                    elif self.current_state == RobotState.APPROACHING:
                        if abs(horizontal_error) > self.align_threshold * 2:
                            self.align_bad_count += 1
                            if self.align_bad_count >= self.align_bad_required:
                                self.align_ok_count = 0
                                self.current_state = RobotState.ALIGNING
                        elif depth_valid and abs(distance_error) < self.distance_threshold:
                            self.current_state = RobotState.REACHED
                        else:
                            cmd.angular.z = self.compute_angular_cmd(horizontal_error, scale=0.5)
                            if not depth_valid:
                                cmd.linear.x = 0.0  
                            else:
                                distance_error = max(-self.max_distance_error, min(self.max_distance_error, distance_error))
                                self.distance_error_filt = (
                                    self.dist_alpha * distance_error
                                    + (1.0 - self.dist_alpha) * self.distance_error_filt
                                )
                                if self.distance_error_filt > 0:
                                    cmd.linear.x = self.linear_kp * self.distance_error_filt
                                    cmd.linear.x = max(0, min(self.max_linear_speed, cmd.linear.x))
                                else:
                                    cmd.linear.x = self.linear_kp * self.distance_error_filt
                                    cmd.linear.x = max(-self.max_linear_speed * 0.5, min(0, cmd.linear.x))
                    
                    elif self.current_state == RobotState.REACHED:
                        if abs(horizontal_error) > self.align_threshold:
                            cmd.angular.z = self.compute_angular_cmd(horizontal_error, scale=0.3)
                        if depth_valid and abs(distance_error) > self.distance_threshold: 
                            self.current_state = RobotState.APPROACHING
                        cmd.linear.x = 0.0
                    
                    # 绘制显示信息
                    state_text = f"State: {self.current_state.name}"
                    coord_text = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}"
                    error_text = f"H_Err:{horizontal_error:.3f} D_Err:{distance_error:.2f}"
                    
                    cv2.putText(rgb_image, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(rgb_image, coord_text, (cX - 50, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(rgb_image, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    
                    cv2.line(rgb_image, (int(image_center_x) - 20, int(h/2)), (int(image_center_x) + 20, int(h/2)), (0, 255, 255), 2)
                    cv2.line(rgb_image, (int(image_center_x), int(h/2) - 20), (int(image_center_x), int(h/2) + 20), (0, 255, 255), 2)

                    self.prev_horizontal_error = horizontal_error
        
        # 目标丢失处理
        if not target_found:
            self.target_lost_count += 1
            self.target_seen_count = 0
            if self.target_lost_count > self.max_lost_frames:
                if self.current_state != RobotState.SEARCHING:
                    self.current_state = RobotState.SEARCHING
                cmd.angular.z = 0.3
                cmd.linear.x = 0.0

        # 🌟🌟🌟 诊断打印代码开始 🌟🌟🌟
        if not target_found:
            rospy.logwarn(f"[Perception] Target not found (Not detected by {'YOLO' if self.detect_mode == 'yolo' else 'Color'})")
        else:
            rospy.loginfo(f"[Perception] Target locked! 2D Coords: cX={cX}, cY={cY}")
            if self.latest_depth is None:
                rospy.logerr("[Perception] FATAL: Depth image missing! No distance data!")
            else:
                if X is not None and Y is not None and Z is not None:
                    # Output 3D coordinates and depth with 3 decimal places
                    rospy.loginfo(f"[Perception] Depth OK! 3D Coords: X={X:.3f}m, Y={Y:.3f}m, Z(Dist)={Z:.3f}m")
                else:
                    rospy.logwarn("[Perception] Warning: Invalid 3D coords in current frame (blind spot or reflection)")
        
        rospy.loginfo(f"[State Machine] Current State: {self.current_state.name}")
        rospy.loginfo(f"[Velocity] linear.x(fwd): {cmd.linear.x:.2f}, angular.z(turn): {cmd.angular.z:.2f}")
        rospy.loginfo("-" * 55) 
        # 🌟🌟🌟 诊断打印代码结束 🌟🌟🌟

        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)

        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)
        
        # 显示画面
        cv2.imshow("Detection Result (RGB)", self.resize(rgb_image))
        if self.detect_mode == 'color' and hsv_result is not None and edges is not None:
            cv2.imshow("HSV Result", self.resize(hsv_result))
            cv2.imshow("Canny Edges", self.resize(edges))
        if self.depth_colormap is not None:
            cv2.imshow("Depth Colormap", self.resize(self.depth_colormap))
        
        cv2.waitKey(1)

    def resize(self,image):
        self.display_scale = rospy.get_param("~display_scale", 0.5)
        return cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

    def compute_angular_cmd(self, horizontal_error, scale=1.0):
        # 过零保护：误差方向翻转时立即停，避免摆头过冲
        if self.prev_horizontal_error is not None:
            if horizontal_error * self.prev_horizontal_error < 0 and abs(horizontal_error) < self.angular_fast_threshold:
                return 0.0

        err = abs(horizontal_error)

        # 小误差死区
        if err < self.angular_slow_threshold or err < self.angular_deadband:
            return 0.0

        # 中误差区：慢速微调（不启用最小角速度）
        if err < self.angular_fast_threshold:
            cmd = -self.angular_kp * 0.5 * scale * horizontal_error
            max_slow = self.max_angular_speed * 0.4
            return max(-max_slow, min(max_slow, cmd))

        # 大误差区：快速转向（启用最小角速度）
        cmd = -self.angular_kp * scale * horizontal_error
        if abs(cmd) < self.min_angular_speed:
            cmd = self.min_angular_speed * (1 if cmd > 0 else -1)
        return max(-self.max_angular_speed, min(self.max_angular_speed, cmd))

    def calculate_3d_coordinates(self, u, v, depth_image):
        """
        基于 ROI (Region of Interest) 中位数滤波的三维坐标计算
        不再只取1个像素点，而是取中心周围的区域过滤噪点，极大提升测距稳定性
        """
        # 1. 确定相机内参
        if self.fx is None:
            fx, fy, cx, cy = 554.25, 554.25, 320.5, 240.5
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        # 2. 获取深度图的尺寸，防止边界溢出报错
        h, w = depth_image.shape

        # 3. 设定 ROI 区域大小，这里取目标中心点周围 20x20 像素的范围
        box_size = 10  # 半径为 10 像素
        u_min = max(0, u - box_size)
        u_max = min(w - 1, u + box_size)
        v_min = max(0, v - box_size)
        v_max = min(h - 1, v + box_size)

        # 4. 提取该小方块区域内的所有深度数据
        roi = depth_image[v_min:v_max, u_min:u_max]

        # 5. 核心抗噪：过滤掉区域里的无效深度值 (比如背景的0，或者反光造成的 NaN)
        valid_depths = roi[(roi > 0.0) & (~np.isnan(roi))]

        # 6. 取有效深度的“中位数”作为最终的 Z 值
        if len(valid_depths) > 0:
            Z = float(np.median(valid_depths))
            
            # 【新增这几行】：单位换算。如果 Z 的数值大得离谱（比如大于10），说明它的单位是毫米
            # 需要除以 1000 转换为米，与你设置的 desired_distance (0.5米) 匹配
            if Z > 10.0:
                Z = Z / 1000.0
                
        else:
            # 如果整个 20x20 的区域都测不到深度（说明真的在盲区），为了安全返回 0.0
            return 0.0, 0.0, 0.0

        # 7. 根据针孔相机模型反推真实 3D 物理坐标
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
            if not self.tf_buffer.can_transform("odom", point_cam.header.frame_id, rospy.Time(0), rospy.Duration(0.2)):
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
