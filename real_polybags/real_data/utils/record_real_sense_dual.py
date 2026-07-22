import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import time
from datetime import datetime
import os

class DualRealsenseRecorder:
    def __init__(self, save_dir="recordings"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create timestamp for this recording session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Queues for frame data
        self.frame_queues = {0: queue.Queue(), 1: queue.Queue()}
        self.display_queues = {0: queue.Queue(maxsize=2), 1: queue.Queue(maxsize=2)}
        self.recording = False
        
        # Pipeline and config for each camera
        self.pipelines = []
        self.configs = []
        
        # Latest frames for display
        self.latest_frames = {0: None, 1: None}
        self.frame_lock = threading.Lock()
        
    def initialize_cameras(self, width=640, height=480, fps=30):
        """Initialize both RealSense cameras"""
        # Get connected devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) < 2:
            raise RuntimeError(f"Found only {len(devices)} camera(s). Need 2 cameras.")
        
        print(f"Found {len(devices)} RealSense cameras")
        
        for i, device in enumerate(devices[:2]):
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"Camera {i}: Serial number {serial}")
            
            # Start pipeline — retry with backoff, since "failed to set power
            # state" on macOS is frequently an intermittent USB timing issue
            # when a second RealSense device claims the bus shortly after the
            # first (known upstream librealsense/macOS bug, worse over a
            # shared USB hub than separate ports).
            #
            # IMPORTANT: a fresh pipeline + config must be created for every
            # attempt. Reusing the same pipeline object after a failed
            # start() leaves the USB handle half-claimed, causing the next
            # attempt to fail with "UVC device is already opened!" instead of
            # actually retrying cleanly.
            pipeline = None
            profile  = None
            last_err = None
            for attempt in range(4):
                pipeline = rs.pipeline()
                config   = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                try:
                    profile = pipeline.start(config)
                    break
                except RuntimeError as e:
                    last_err = e
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    wait_s = 2.0 * (attempt + 1)
                    print(f"Camera {i}: pipeline.start() failed ({e}), "
                          f"retrying in {wait_s:.1f}s (attempt {attempt + 1}/4)")
                    time.sleep(wait_s)
            if profile is None:
                raise RuntimeError(
                    f"Camera {i}: pipeline.start() failed after 4 attempts: {last_err}"
                )

            # Give the USB bus a moment to settle before the next camera
            # claims it — this is the delay that was previously missing
            # entirely between camera 0 and camera 1.
            time.sleep(1.5)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"Camera {i} depth scale: {depth_scale}")
            
            self.pipelines.append(pipeline)
            self.configs.append(config)
            
        # Let cameras warm up
        time.sleep(2)
        
    def capture_frames(self, camera_idx):
        """Capture frames from a specific camera"""
        pipeline = self.pipelines[camera_idx]
        frame_count = 0
        
        while self.recording:
            try:
                # Wait for frames
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                
                # Get color and depth frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Get timestamp
                timestamp = frames.get_timestamp()
                
                frame_data = {
                    'color': color_image.copy(),
                    'depth': depth_image.copy(),
                    'timestamp': timestamp,
                    'frame_num': frame_count
                }
                
                # Put in save queue
                self.frame_queues[camera_idx].put(frame_data)
                
                # Update latest frame for display (non-blocking)
                with self.frame_lock:
                    self.latest_frames[camera_idx] = frame_data
                
                frame_count += 1
                
            except Exception as e:
                print(f"Error capturing from camera {camera_idx}: {e}")
                
    def save_frames(self, camera_idx):
        """Save frames from queue to disk"""
        cam_dir = os.path.join(self.save_dir, f"camera_{camera_idx}_{self.timestamp}")
        color_dir = os.path.join(cam_dir, "color")
        depth_dir = os.path.join(cam_dir, "depth")
        
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        # Save metadata
        metadata_file = os.path.join(cam_dir, "metadata.txt")
        
        saved_count = 0
        
        while self.recording or not self.frame_queues[camera_idx].empty():
            try:
                frame_data = self.frame_queues[camera_idx].get(timeout=1)
                
                frame_num = frame_data['frame_num']
                
                # Save color image
                color_path = os.path.join(color_dir, f"color_{frame_num:06d}.png")
                cv2.imwrite(color_path, frame_data['color'])
                
                # Save depth image
                depth_path = os.path.join(depth_dir, f"depth_{frame_num:06d}.png")
                cv2.imwrite(depth_path, frame_data['depth'])
                
                # Save metadata
                with open(metadata_file, 'a') as f:
                    f.write(f"{frame_num},{frame_data['timestamp']}\n")
                
                saved_count += 1
                if saved_count % 30 == 0:  # Print every 30 frames (1 second at 30fps)
                    print(f"Camera {camera_idx}: Saved {saved_count} frames")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving from camera {camera_idx}: {e}")
        
        print(f"Camera {camera_idx}: Total saved {saved_count} frames")
    
    def start_recording(self):
        """Start recording from both cameras"""
        self.recording = True
        
        # Start capture threads
        self.capture_threads = []
        for i in range(2):
            t = threading.Thread(target=self.capture_frames, args=(i,), daemon=True)
            t.start()
            self.capture_threads.append(t)
        
        # Start save threads
        self.save_threads = []
        for i in range(2):
            t = threading.Thread(target=self.save_frames, args=(i,), daemon=True)
            t.start()
            self.save_threads.append(t)
            
        print("Recording started. Press 'q' to stop.")
    
    def stop_recording(self):
        """Stop recording"""
        print("\nStopping recording...")
        self.recording = False
        
        # Wait for threads to finish
        print("Waiting for capture threads...")
        for t in self.capture_threads:
            t.join(timeout=5)
        
        print("Waiting for save threads...")
        for t in self.save_threads:
            t.join(timeout=10)
            
        print("Recording stopped.")
    
    def cleanup(self):
        """Stop pipelines and cleanup"""
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except:
                pass
        print("Cleanup complete.")
    
    def display_preview(self):
        """Display live preview from both cameras"""
        print("Display window opened. Press 'q' to stop recording.")
        
        while self.recording:
            try:
                # Get latest frames (thread-safe)
                with self.frame_lock:
                    frames = [self.latest_frames[0], self.latest_frames[1]]
                
                if frames[0] is not None and frames[1] is not None:
                    # Create depth colormap
                    depth_colormap_0 = cv2.applyColorMap(
                        cv2.convertScaleAbs(frames[0]['depth'], alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    depth_colormap_1 = cv2.applyColorMap(
                        cv2.convertScaleAbs(frames[1]['depth'], alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    
                    # Stack images
                    top_row = np.hstack((frames[0]['color'], depth_colormap_0))
                    bottom_row = np.hstack((frames[1]['color'], depth_colormap_1))
                    combined = np.vstack((top_row, bottom_row))
                    
                    # Add labels and frame info
                    cv2.putText(combined, f"Camera 0 - Color (Frame: {frames[0]['frame_num']})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, "Camera 0 - Depth", 
                               (frames[0]['color'].shape[1] + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, f"Camera 1 - Color (Frame: {frames[1]['frame_num']})", 
                               (10, frames[0]['color'].shape[0] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, "Camera 1 - Depth", 
                               (frames[0]['color'].shape[1] + 10, frames[0]['color'].shape[0] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add recording indicator
                    cv2.circle(combined, (combined.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(combined, "REC", (combined.shape[1] - 80, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow('Dual RealSense Recording', combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_recording()
                    break
                    
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(0.01)
                continue
        
        cv2.destroyAllWindows()

def main():
    recorder = DualRealsenseRecorder(save_dir="dual_camera_recordings_11_2_2026")
    
    try:
        # Initialize cameras
        print("Initializing cameras...")
        recorder.initialize_cameras(width=640, height=480, fps=30)
        
        # Start recording
        recorder.start_recording()
        
        # Display preview (blocks until 'q' is pressed)
        recorder.display_preview()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main()