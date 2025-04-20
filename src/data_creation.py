import os
import subprocess


def spl(input_dir):
    segment_duration = 20  # in seconds
    # Make output directory
    output_dir = os.path.join(input_dir, "segments")
    os.makedirs(output_dir, exist_ok=True)
    # Supported extensions
    supported_extensions = [".wav", ".m4a"]

    def split_audio(file_path, output_dir, segment_duration):
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        output_pattern = os.path.join(output_dir, f"{name}_part%02d{ext}")
        # Run ffmpeg to split the file
        cmd = [
            "ffmpeg",
            "-i", file_path,
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-c", "copy",
            output_pattern
        ]
        print(f"Splitting {filename}...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Finished splitting {filename}")

    for fname in os.listdir(input_dir):
        if any(fname.lower().endswith(ext) for ext in supported_extensions):
            full_path = os.path.join(input_dir, fname)
            split_audio(full_path, output_dir, segment_duration)

    print("Done splitting all files.")


spl("./Training_Data/Yaman-discard/Yaman-wav/")
