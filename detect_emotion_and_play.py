#open the webcam record the video,
#when user closed
#save video from webcamp
import imageio as iio
import matplotlib.pyplot as plt
import time
#start process ...to continue with the next cell
camera = iio.get_reader("<video0>")
meta = camera.get_meta_data()
num_frames = 5 * int(meta["fps"])
delay = 1/meta["fps"]

buffer = list()
for frame_counter in range(num_frames):
    frame = camera.get_next_data()
    buffer.append(frame)
    time.sleep(delay)

camera.close()

iio.mimwrite("frame.mp4", buffer, macro_block_size=8, fps=meta["fps"])
#saved file in given path

##################################################################################
#Fetch emotion
from fer import Video
from fer import FER
import os
import sys
import pandas as pd
import matplotlib as plt
from time import sleep

location_videofile = "frame.mp4"
# But the Face detection detector
face_detector = FER(mtcnn=True)
# Input the video for processing
input_video = Video(location_videofile)

processing_data = input_video.analyze(face_detector, display=False)

vid_df = input_video.to_pandas(processing_data)
vid_df = input_video.get_first_face(vid_df)
vid_df = input_video.get_emotions(vid_df)

# Plotting the emotions against time in the video
pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()

angry = sum(vid_df.angry)
disgust = sum(vid_df.disgust)
fear = sum(vid_df.fear)
happy = sum(vid_df.happy)
sad = sum(vid_df.sad)
surprise = sum(vid_df.surprise)
neutral = sum(vid_df.neutral)

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
score_comparisons['Emotion Value from the Video'] = emotions_values
print(score_comparisons)

final_data = score_comparisons.sort_values(by=['Emotion Value from the Video'],ascending=False)

import os 
os.remove('E:/project code/play_music_face_detection/Output/images.zip')

print(final_data)
sorted_data = list(final_data.iloc[0])
print(sorted_data)
detected_emotion = str(sorted_data[0])
print("Detected Emtion: ",detected_emotion)
#fill all song folders by music before test
####################################################################################
#play song Module
if detected_emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
    song = "song1.wav"
def freeze_support():
    # import multiprocessing
    from playsound import playsound
    playsound("E:\\project code\\play_music_face_detection\\code\\music\\"+detected_emotion+"\\"+song)
    # p = multiprocessing.Process(target=playsound, args=("E:\\BE projects\\music prediction\\code\\music\\"+detected_emotion+"\\"+song,))
    # p.start()
    # input("press ENTER to stop song")
    # p.terminate()

# if __name__ == '__main__':
freeze_support()


