import cv2
import os
import glob

# Settings
input_folder = "frames"          # Folder containing images
output_folder = "output_videos"  # Folder to save videos
image_extension = "*.jpg"        # Change to *.png if needed
frame_rates = [5, 10, 15, 20, 25]

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Read all images from folder (sorted)
image_files = sorted(glob.glob(os.path.join(input_folder, image_extension)))

if not image_files:
    print("❌ No images found in folder!")
else:
    print(f"✅ Found {len(image_files)} images")

    # Read first image to get frame size
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape
    print(f"📐 Frame size: {width}x{height}")

    # Loop through each frame rate
    for fps in frame_rates:
        output_path = os.path.join(output_folder, f"output_{fps}fps.avi")

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"\n🎬 Writing video at {fps} FPS → {output_path}")

        for i, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"  ⚠️ Skipping unreadable image: {img_path}")
                continue

            writer.write(frame)

            # Progress update every 100 frames
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} frames...")

        writer.release()
        print(f"  ✅ Saved: {output_path}")

    print("\n🎉 All videos created successfully!")