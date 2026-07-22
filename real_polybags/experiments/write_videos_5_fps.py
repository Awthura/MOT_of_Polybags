import cv2
import os
import glob
import threading
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# ✅ Set your video folder path here
VIDEO_FOLDER = r"C:\Users\spsata\Desktop\MoPeFf-KIDZ\Mult_Camera_MOT_30_3_2026\Experiments_28_05_2026\Single_Polybags\Iteration_1"

CAMERA_SOURCES = [
    {"id": os.path.join(VIDEO_FOLDER, "basler_1_1280x720_20260528_091126.avi"),  "name": "Camera_1"},
    {"id": os.path.join(VIDEO_FOLDER, "basler_2_1280x720_20260528_091126.avi"),  "name": "Camera_2"},
    {"id": os.path.join(VIDEO_FOLDER, "lucid_1280x720_20260528_091126.avi"),     "name": "Camera_3"},
    {"id": os.path.join(VIDEO_FOLDER, "rgbd_1_color_1280x720_20260528_091126.avi"), "name": "Camera_4"},
    {"id": os.path.join(VIDEO_FOLDER, "rgbd_2_color_1280x720_20260528_091126.avi"), "name": "Camera_5"},
]

OUTPUT_BASE_FOLDER = os.path.join(VIDEO_FOLDER, "output")
OUTPUT_VIDEO_FPS   = 5
IMAGE_FORMAT       = "jpg"
MAX_FRAMES         = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_folders(camera_name, base_folder):
    image_folder = os.path.join(base_folder, camera_name, "images")
    video_folder = os.path.join(base_folder, camera_name, "videos")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    return image_folder, video_folder


def get_video_fps(cap, cam_name):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[{cam_name}] ⚠️  Could not read FPS, defaulting to 25")
        fps = 25
    print(f"[{cam_name}] 🎯 Auto-detected FPS: {fps}")
    return fps


def extract_frames(camera, base_folder, results):
    cam_id   = camera["id"]
    cam_name = camera["name"]

    # ✅ Check if file exists before opening
    if not os.path.exists(cam_id):
        print(f"[{cam_name}] ❌ File not found: {cam_id}")
        results[cam_name] = {"success": False, "frames": 0}
        return

    print(f"[{cam_name}] 🎥 Opening: {cam_id}")
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"[{cam_name}] ❌ Failed to open: {cam_id}")
        results[cam_name] = {"success": False, "frames": 0}
        return

    # Auto-read FPS from video
    original_fps = get_video_fps(cap, cam_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / original_fps if original_fps > 0 else 0

    print(f"[{cam_name}] 📊 Total Frames : {total_frames}")
    print(f"[{cam_name}] 📐 Resolution   : {width}x{height}")
    print(f"[{cam_name}] ⏱️  Duration     : {duration:.2f} seconds")

    frame_interval = max(1, round(original_fps / OUTPUT_VIDEO_FPS))
    print(f"[{cam_name}] 🔢 Frame Interval: every {frame_interval} frame(s) "
          f"({original_fps} → {OUTPUT_VIDEO_FPS} FPS)")

    image_folder, video_folder = create_folders(cam_name, base_folder)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if MAX_FRAMES and saved_count >= MAX_FRAMES:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(image_folder, f"frame_{saved_count:06d}.{IMAGE_FORMAT}")
            cv2.imwrite(filename, frame)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"[{cam_name}] 💾 Saved {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"[{cam_name}] ✅ Done! Saved {saved_count} images → {image_folder}")

    results[cam_name] = {
        "success"      : True,
        "frames"       : saved_count,
        "original_fps" : original_fps,
        "image_folder" : image_folder,
        "video_folder" : video_folder,
        "width"        : width,
        "height"       : height,
    }


def create_video_from_images(cam_name, image_folder, video_folder, width, height, fps=5):
    image_files = sorted(glob.glob(os.path.join(image_folder, f"*.{IMAGE_FORMAT}")))

    if not image_files:
        print(f"[{cam_name}] ❌ No images found in {image_folder}")
        return

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(video_folder, f"{cam_name}_{fps}fps_{timestamp}.avi")
    fourcc      = cv2.VideoWriter_fourcc(*'XVID')
    writer      = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[{cam_name}] 🎬 Creating video at {fps} FPS → {output_path}")

    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{cam_name}] ⚠️  Skipping unreadable: {img_path}")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"[{cam_name}] 🖼️  Written {i + 1}/{len(image_files)} frames...")

    writer.release()
    print(f"[{cam_name}] ✅ Video saved → {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Multi-Camera Frame Extractor & Video Writer")
    print("=" * 60)

    # ✅ Verify all video files exist before starting
    print("\n🔍 Checking video files...\n")
    for camera in CAMERA_SOURCES:
        exists = os.path.exists(camera["id"])
        status = "✅ Found" if exists else "❌ Not Found"
        print(f"  {status} → {camera['id']}")

    print()
    start_time = datetime.now()
    results    = {}
    threads    = []

    # Step 1: Extract frames
    print("\n📷 Step 1: Extracting frames from all cameras...\n")
    for camera in CAMERA_SOURCES:
        t = threading.Thread(
            target=extract_frames,
            args=(camera, OUTPUT_BASE_FOLDER, results)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Step 2: Create videos
    print("\n🎬 Step 2: Creating videos at 5 FPS...\n")
    for cam_name, info in results.items():
        if info["success"]:
            create_video_from_images(
                cam_name     = cam_name,
                image_folder = info["image_folder"],
                video_folder = info["video_folder"],
                width        = info["width"],
                height       = info["height"],
                fps          = OUTPUT_VIDEO_FPS
            )

    # Summary
    end_time = datetime.now()
    elapsed  = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("  ✅ SUMMARY")
    print("=" * 60)
    for cam_name, info in results.items():
        status = "✅ Success" if info["success"] else "❌ Failed"
        frames = info.get("frames", 0)
        orig   = info.get("original_fps", "N/A")
        print(f"  {cam_name}: {status} | Original FPS: {orig} | Saved Frames: {frames}")

    print(f"\n  ⏱️  Total Time: {elapsed:.2f} seconds")
    print(f"  📁 Output Folder: {os.path.abspath(OUTPUT_BASE_FOLDER)}")
    print("=" * 60)


if __name__ == "__main__":
    main()