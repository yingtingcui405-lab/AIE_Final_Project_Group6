#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
import pyaudio
import numpy as np
import time
from scipy.signal import correlate
import matplotlib.pyplot as plt

# ==================== 【Configuration】 ====================
DEVICE_INDEX = 5           # ASTRA Pro Default device number
CHUNK = 2048               # Sampling points per frame
RATE = 48000               # Sampling rate（Hz）
TRIGGER_VOL = 400          # Trigger volume threshold
DISTANCE_THRESHOLD = 0.02  # Minimum distance difference for direction judgment (unit: meters, 2cm is sufficient)
MIC_DISTANCE = 0.12       # Distance between two microphones (unit: meters, recommended 10~20cm)

SPEED_OF_SOUND = 343.0     # Speed of sound (m/s)
TURN_SPEED = 1.0         # Turning angular velocity (rad/s)
FORWARD_SPEED = 1.5      # Forward linear velocity (m/s)
GO_TIME = 1.5              # Forward duration (seconds)
CONFIRM_COUNT = 2          # Continuous same direction count to execute action (anti-jitter)

# Whether to enable real-time visualization (for debugging, will pop up window during runtime; set to False in production environment)
ENABLE_PLOT = True
# =====================================================================

rospy.init_node("tdoa_sound_locator", anonymous=True)
cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
time.sleep(0.5)

# Initialize microphone (must be dual-channel)
p = pyaudio.PyAudio()
try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )
except Exception as e:
    rospy.logerr(f"Microphone opening failed! Please check DEVICE_INDEX={DEVICE_INDEX} is correct. Error: {e}")
    exit(1)

def stop():
    twist = Twist()
    cmd_pub.publish(twist)

def move_robot(linear_x=0.0, angular_z=0.0):
    twist = Twist()
    twist.linear.x = linear_x
    twist.angular.z = angular_z
    cmd_pub.publish(twist)

# ==================== Core: GCC-PHAT Sound Source Localization ====================
def get_tdoa_direction(data):
    """Calculate the sound source direction using the GCC-PHAT algorithm"""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    left = samples[0::2]
    right = samples[1::2]

    volume = np.max(np.abs(samples))
    if volume < TRIGGER_VOL:
        return "NONE", 0.0, volume

    # === GCC-PHAT algorithm ===
    N = len(left)
    # 1. FFT to frequency domain
    fft_left = np.fft.rfft(left)
    fft_right = np.fft.rfft(right)

    # 2. Calculate cross-power spectrum G_xy(f) = X(f) * conj(Y(f))
    cross_spectrum = fft_left * np.conj(fft_right)

    # 3. PHAT weighting：G_phate(f) = G_xy(f) / |G_xy(f)| （Phase only）
    eps = 1e-10
    phat_weight = cross_spectrum / (np.abs(cross_spectrum) + eps)

    # 4. IFFT back to time domain → obtain GCC-PHAT cross-correlation function
    gcc_phat = np.fft.irfft(phat_weight, n=N)

    # 5. Find peak position (Note: rfft's center is N//2)
    lag = np.argmax(gcc_phat)
    center = N // 2
    tdoa_samples = lag - center

    # 6. Convert to physical quantity
    tdoa_seconds = tdoa_samples / RATE
    distance_diff = tdoa_seconds * SPEED_OF_SOUND

    # 7. Direction judgment
    direction = "FRONT"
    if abs(distance_diff) > DISTANCE_THRESHOLD:
        if distance_diff > 0:
            direction = "LEFT"
        else:
            direction = "RIGHT"

    return direction, distance_diff, volume

# ==================== Visualization (for debugging) ====================
fig, ax = None, None
line = None
if ENABLE_PLOT:
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    line, = ax.plot([], [], 'b-', linewidth=1.5)
    ax.set_xlim(-CHUNK//2, CHUNK//2)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Lag (samples)')
    ax.set_title('GCC-PHAT Correlation (Real-time)')
    ax.grid(True)

# ==========================================================

rospy.loginfo("="*70)
rospy.loginfo("TDOA Sound Source Localization Controller Started!")
rospy.loginfo(f"Microphone: {DEVICE_INDEX}, Spacing: {MIC_DISTANCE:.2f}m, Sample Rate: {RATE}Hz")
rospy.loginfo("Clap Test Recommendation: Stand at 30°~60° to the robot's side, Distance: 0.5~1.5m")
rospy.loginfo("Hint: If FRONT is always displayed, try increasing the angle or decreasing DISTANCE_THRESHOLD")
rospy.loginfo("="*70)

# Direction confirmation queue
direction_queue = []

try:
    while not rospy.is_shutdown():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except OSError as e:
            rospy.logwarn(f"Audio reading exception: {e}，skipping...")
            continue

        direction, dist_diff, volume = get_tdoa_direction(data)

        # 【Visualization】update plot
        if ENABLE_PLOT and fig:
            try:
                # Recalculate GCC-PHAT for plotting (avoid reusing previous variables)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                left = samples[0::2]
                right = samples[1::2]
                N = len(left)
                fft_l = np.fft.rfft(left)
                fft_r = np.fft.rfft(right)
                cs = fft_l * np.conj(fft_r)
                phat = cs / (np.abs(cs) + 1e-10)
                gcc = np.fft.irfft(phat, n=N)
                x_axis = np.arange(-N//2, N//2)
                line.set_data(x_axis, gcc)
                ax.set_ylim(-1, 1)
                fig.canvas.draw()
                fig.canvas.flush_events()
            except Exception as e:
                rospy.logdebug(f"Plotting exception: {e}")

        # Process only when volume is sufficiently high
        if direction != "NONE":
            rospy.loginfo(f"Volume: {int(volume):4d} | Direction: {direction:>6} | ΔDistance: {dist_diff:+.3f}m")

            # Direction confirmation queue (debounce)
            direction_queue.append(direction)
            if len(direction_queue) > CONFIRM_COUNT:
                direction_queue.pop(0)

            # When all directions in the queue are consistent, execute
            if len(direction_queue) == CONFIRM_COUNT and len(set(direction_queue)) == 1:
                final_dir = direction_queue[0]
                rospy.loginfo(f"确认方向: {final_dir} (连续 {CONFIRM_COUNT} 次)")

                # Perform steering
                if final_dir == "LEFT":
                    move_robot(angular_z=TURN_SPEED)
                    time.sleep(0.35)
                elif final_dir == "RIGHT":
                    move_robot(angular_z=-TURN_SPEED)
                    time.sleep(0.35)
                # FRONT No steering, move forward directly

                # Stop with buffer after steering
                stop()
                time.sleep(0.15)

                # Move forward
                move_robot(linear_x=FORWARD_SPEED)
                time.sleep(GO_TIME)
                stop()

                rospy.loginfo("Complete one tracking cycle and wait for the next sound....\n")
                time.sleep(1.0)  # Cooling period
                direction_queue.clear()  # Clear the queue

        # Small delay to avoid CPU saturation
        time.sleep(0.01)

except KeyboardInterrupt:
    rospy.loginfo("User interrupted the program")

finally:
    stop()
    stream.stop_stream()
    stream.close()
    p.terminate()
    if ENABLE_PLOT and fig:
        plt.close(fig)
    rospy.loginfo("\nTDOA Sound Source Localization has been safely stopped")