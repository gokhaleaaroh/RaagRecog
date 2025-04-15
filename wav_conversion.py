import os
import subprocess


def conv(input_dir, output_dir):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Supported input extensions
    input_extensions = ['.m4a', '.opus']
    for filename in os.listdir(input_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in input_extensions:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, name + '.wav')
            print(f'Converting {filename} -> {name}.wav')
            subprocess.run([
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-ar', '44100',
                output_path
            ])


conv('./Training_Data/Kalavati/', './Training_Data/Kalavati/Kalavati-wav/')

conv('./Training_Data/Marwa/', './Training_Data/Marwa/Marwa-wav/')
