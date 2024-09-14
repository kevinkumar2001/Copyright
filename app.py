import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from face_recognition import face_locations
import random

def modify_video(input_video, output_video):
    # Load the video
    clip = VideoFileClip(input_video)
    
    # Modify video frames
    modified_clip = clip.fl_image(modify_frame)
    
    # Modify audio using MoviePy
    modified_audio = modify_audio(clip.audio)
    
    # Set the modified audio to the video
    modified_clip = modified_clip.set_audio(modified_audio)
    
    # Resize video
    new_size = (random.randint(800, 1200), random.randint(600, 900))
    modified_clip = modified_clip.resize(new_size)
    
    # Add subtle random zoom
    modified_clip = modified_clip.fx(lambda t: clip.fx(lambda f: f.resize(1 + 0.05 * np.sin(t)), t))
    
    # Write the result
    modified_clip.write_videofile(output_video, codec='libx264')

def modify_frame(frame):
    # Convert to grayscale and back to color
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Apply subtle blur
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Adjust brightness and contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Modify face structure for bypass using advanced techniques
    face_locs = face_locations(frame)
    for top, right, bottom, left in face_locs:
        face = frame[top:bottom, left:right]
        
        # 1. Pixel Shuffling
        h, w, _ = face.shape
        pixels = face.reshape(-1, 3)
        np.random.shuffle(pixels)
        shuffled_face = pixels.reshape(h, w, 3)
        
        # 2. Warping using random affine transformations
        rows, cols, ch = shuffled_face.shape
        random_shift = np.float32([[1, 0, random.randint(-5, 5)], [0, 1, random.randint(-5, 5)]])
        warped_face = cv2.warpAffine(shuffled_face, random_shift, (cols, rows))
        
        # Replace original face region with the altered one
        frame[top:bottom, left:right] = warped_face
    
    return frame

def modify_audio(audio):
    # Use MoviePy to manipulate audio data efficiently
    audio_data = audio.to_soundarray()
    
    # Apply pitch shift by multiplying frequencies randomly
    pitch_shift_factor = random.uniform(0.8, 1.2)
    modified_audio = audio.fx(lambda sound: sound.speedx(pitch_shift_factor))
    
    # Add background noise using MoviePy's methods
    noise = AudioFileClip("background_noise.wav")
    modified_audio = CompositeAudioClip([modified_audio.volumex(0.8), noise.volumex(0.2)])
    
    return modified_audio

def add_new_audio(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    
    # Ensure new audio matches video duration
    if new_audio.duration > video.duration:
        new_audio = new_audio.subclip(0, video.duration)
    else:
        new_audio = new_audio.fx(vfx.loop, duration=video.duration)
    
    final_audio = CompositeAudioClip([video.audio, new_audio])
    final_video = video.set_audio(final_audio)
    
    final_video.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_modified.mp4"
    new_audio = "new_audio.mp3"
    final_output = "final_output.mp4"
    
    modify_video(input_video, output_video)
    add_new_audio(output_video, new_audio, final_output)
    print("Video processing complete!")
