import base64
import csv
from flask import abort
import glob
import os
import subprocess
import math
import shelve
import re
from sklearn import tree
from sklearn.externals import joblib
from flask_socketio import emit
from lie_to_me import basedir, FFMPEG_PATH, FFPROBE_PATH, app, socketio
from lie_to_me.modules import audio

frames_dir = os.path.join(basedir, 'static', 'data', 'tmp_video')
audio_dir = os.path.join(basedir, 'static', 'data', 'tmp_audio')
json_path = os.path.join(basedir, 'static', 'data', 'tmp_json')
base64_frames = {}
capture_fps = 20  # By default capture 20 frames per second
video_fps_rate = [-1]  # Video FPS Rate


def convert_to_frames(filepath):
    global capture_fps

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if not os.path.exists(json_path):
        os.makedirs(json_path)

    output_filename = os.path.join(frames_dir, "thumb%09d.jpg")  # output filename

    try:
        # FIND DIMENSION OF VIDEO
        vid_dimension_cmd = [FFPROBE_PATH, '-v', 'error', '-show_entries', 'stream=width,height', '-of',
                             'default=noprint_wrappers=1', filepath]
        proc = subprocess.Popen(vid_dimension_cmd, stdout=subprocess.PIPE)
        dimension_output = proc.stdout.read().decode('utf-8')
        regex = re.compile(r"[a-z]+=([0-9]+)\r*\n[a-z]+=([0-9]+)\r*\n")
        width, height = regex.match(dimension_output).groups()

        # FIND FPS RATE OF VIDEO
        fps_rate_cmd = [FFPROBE_PATH, filepath, "-v", "0", "-select_streams", "v", "-print_format", "flat",
                        "-show_entries", "stream=r_frame_rate"]
        fps_output = subprocess.check_output(fps_rate_cmd).decode('utf-8')
        fps_list = fps_output.split('=')[1].strip()[1:-1].split('/')

        fps_rate = float(fps_list[0]) if len(fps_list) == 1 else float(fps_list[0]) / float(fps_list[1])

        # SPLIT VIDEO TO FRAMES
        capture_fps = min(capture_fps, int(fps_rate))  # ensure fps rate isn't exceeded
        frames_split_cmd = [FFMPEG_PATH, '-i', filepath, '-r', str(capture_fps), output_filename, '-hide_banner']
        subprocess.call(frames_split_cmd)  # break video to its frames

        return width, height, capture_fps
    except Exception as e:
        print(e)
        return abort(404)


def convert_audio(filepath):
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    output = os.path.join(audio_dir, "%03d.wav")

    try:
        ffmpeg_command = [FFMPEG_PATH, '-i', filepath, '-ar', '11000', '-ac', '2',
                          '-f', 'segment', '-segment_time', '2', output]
        subprocess.call(ffmpeg_command)  # convert video into wave file

        files = [(audio_dir + '/' + f) for f in os.listdir(audio_dir)]
        return files

    except Exception as e:
        print(e)
        return abort(404)


def process_video(filepath):
    """
        Processes Video Submitted by User
    """
    global video_fps_rate

    width, height, fps_rate = convert_to_frames(filepath)  # convert the video to images
    ordered_files = sorted(os.listdir(frames_dir), key=lambda x: (int(re.sub(r'\D','',x)),x))

    # Convert all frames to base64 images and begin calling
    for index, frame in enumerate(ordered_files):
        with open(os.path.join(frames_dir, frame), 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
            base64_frames[index] = encoded_string.decode('utf-8')

    cleanup_video_frames()

    video_fps_rate[0] = fps_rate

    # Frames are ready - start sending them to for pooling
    # Let's emit a message indicating that we're about to start sending files
    socketio.emit('canvas_width_height', {'width': width, 'height': height})


def process_audio(filepath):
    """ Process Audio component of Video
    """
    json_path = os.path.join(basedir, 'static', 'data', 'tmp_json')

    mean_energy = []
    max_pitch_amp = []
    vowel_duration = []
    pitch_contour = []

    output = convert_audio(filepath)

    for files in output:
        frames, framelength = audio.split(files)
        filteredframes = audio.applyhamming(frames)
        energy = audio.energy(filteredframes)
        fourier = audio.fourier(filteredframes)
        frames = audio.inverse_fourier(fourier)
        pitchamp, pitchperiod = audio.sampling(frames)

        # Implemented Features, read audio.py for return values
        data1 = audio.meanenergy(energy)
        data2 = audio.maxpitchamp(pitchamp)
        data3 = audio.vowelduration(pitchamp, data2)
        data4 = audio.fundamentalf(pitchperiod, framelength)

        mean_energy.append(data1)
        max_pitch_amp.append(data2)
        vowel_duration.append(data3)
        pitch_contour.append(data4)

    with shelve.open(os.path.join(json_path, 'audio_data.shlf')) as shelf:
        shelf['mean_energy'] = mean_energy
        shelf['max_pitch_amp'] = max_pitch_amp
        shelf['vowel_duration'] = vowel_duration
        shelf['pitch_contour'] = pitch_contour

    socketio.emit('data_complete', 'Audio_Complete')

    cleanup_audio()


def detect_blinks(eye_closure_list, fps):
    """
        Returns the frames where blinks occured
    """
    eye_cl_thresh = 50  # eye closure >= 50 to be considered closed
    eye_cl_consec_frames = 1  # 1 or more consecutive frames to be considered a blink
    counter = 0

    # Array of frames where blink occured
    blink_timestamps = []

    # Instantaneous blink rate (blink rate after every 2 secs)
    # blink rate = total number of blinks / time (in minutes) = blinks/minute
    total_blinks = 0
    elapsed_seconds = 0
    two_sec_save = 0
    two_sec_tracker = 0

    for frame_number, eye_thresh in enumerate(eye_closure_list):
        if eye_thresh is None:
            pass
        elif eye_thresh > eye_cl_thresh:
            counter += 1
        else:
            if counter >= eye_cl_consec_frames:
                total_blinks += 1
                # seconds = frame_number / fps
                # minutes = seconds / 60
                # if minutes < 1:
                #     minutes = 0
                # blink_timestamps.append((minutes, seconds))
            counter = 0

        # convert processed frames to number of minutes
        elapsed_seconds = ((frame_number+1) / fps)

        # tracker to see if two secs have passed since blink rate was last captured
        two_sec_tracker = elapsed_seconds - two_sec_save

        # Goal is to capture blink rate every two seconds
        if two_sec_tracker >= 2:
            two_sec_save += two_sec_tracker
            two_sec_tracker = 0
            blink_rate = total_blinks / elapsed_seconds  # in blinks per second
            blink_timestamps.append(blink_rate)

    return blink_timestamps


def microexpression_analyzer(emotions, fps):
    """Micro expressions happen in 1/25th of a second or 15 frames

        Microexpression_analyzer:
        Function parameters:
          **emotions: A dictionary of emotions and their coresponding
          number value from 1 to 100 and the index is
          the current frame e.g emotions[0]['anger']
        Return:
          The function returns a list of timestamps where a
          possible micro expression was detechted
    """
    current_max = 0
    previous_max = 0
    microexpression_loop_counter = 0
    flag = 0
    previous_emotion = ''
    emotion_at_start = ''
    list_of_emotions = ['anger', 'contempt', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    timestamps = []
    seconds_timestamps = []

    for i in range(len(emotions)):
        # Store current max of emotions
        if not emotions[i]:
            continue

        current_max = max(emotions[i]['anger'],
                          emotions[i]['contempt'],
                          emotions[i]['disgust'],
                          emotions[i]['joy'],
                          emotions[i]['sadness'],
                          emotions[i]['surprise'])

        # Store the current emotion
        for key in emotions[i].keys():
            if current_max == emotions[i][key]:
                current_emotion = key

        if i == 0:
            previous_max = current_max
            previous_emotion = current_emotion
            continue

        # If previous_emotion is not equal to current_emotion then reset the counter and emotion_at_start
        if previous_emotion != current_emotion:
            if microexpression_loop_counter != 15:
                microexpression_loop_counter = 0
                microexpression_loop_counter += 1
                emotion_at_start = previous_emotion

        # Checking to see if the expression stayed the same, if so we increment a
        elif previous_emotion == current_emotion:
            microexpression_loop_counter += 1
        # If the micro expression changed back to the original expression it came from then
        # We have a possible lie and the timestamp is recorded.
        if previous_emotion != current_emotion and microexpression_loop_counter == 15:
            seconds = i / fps
            minutes = seconds / 60
            if minutes < 1:
                minutes = 0
            timestamps.append((minutes, seconds))
            seconds_timestamps.append(seconds)
            microexpression_loop_counter = 0
            emotion_at_start = ''
            continue
        # Record current max and previous max for next the analysis of the next ones
        previous_max = current_max
        previous_emotion = current_emotion

    total_seconds = len(emotions) / fps
    time_array = [0]*(math.ceil(total_seconds/2))
    for i in range(len(timestamps)):
        start_seconds = 0
        end_seconds = 120
        count = 0
        for j in range((math.ceil(total_seconds/2))):
            if start_seconds < timestamps[i][1] <= end_seconds:
                time_array[count] += 1
                break
            else:
                count += 1
                start_seconds += 120
                end_seconds += 120

    return time_array

# Function to train support vector model. Only for use when training the model.
def train_lie_model(pkl_file):
    list_of_features = []
    training_data = []
    with open('niko_train.csv', 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            features = []
            training_data.append(row['False/True'])
            features.append(row['Micro-expressions'])
            features.append(row['Blinks'])
            features.append(row['Mean Energy'])
            features.append(row['Max Pitch Amplitude'])
            features.append(row['Vowel Duration'])
            features.append(row['Fundamental Frequency'])
            list_of_features.append(features)

    if not os.path.exists(pkl_file):
        dt = tree.DecisionTreeClassifier()
        dt.fit(list_of_features, training_data)
        print('Model Trained')
        joblib.dump(dt, pkl_file)
    else:
        os.remove(pkl_file)
        dt = tree.DecisionTreeClassifier()
        dt.fit(list_of_features, training_data)
        print('Model Trained')
        joblib.dump(dt, pkl_file)

# Take the model and predicts whether a lie occured
# Arguments
# vector: which is a list of lists of features to be predicted.
# Example [[Micro-expr #1, blink-rate #1, audio-feature #1, ...]
# [Micro-expr #2, blink-rate #2, audio-feature #2, ...] [...] [...]]


def predict(vector):
    """ Machine Learning Lie Prediction Function"""
    pkl_file = os.path.join(basedir, 'MLModels', 'DT_ML_model(Microexpressions).pkl')

    if not os.path.exists(pkl_file):
        exit()

    SVM = joblib.load(pkl_file)
    return SVM.predict(vector)


def cleanup_uploads():
    """Clean up uploaded videos"""
    for fl in glob.glob(os.path.join(basedir, 'static', 'data', 'uploads', "*")):
        os.remove(fl)


def cleanup_video_frames():
    """ Clean up temporary frames and uploaded file
    """
    for fl in glob.glob(os.path.join(basedir, 'static', 'data', 'tmp_video', '*')):
        os.remove(fl)


def cleanup_audio():
    """ Clean up temporary audio files
    """
    for fl in glob.glob(os.path.join(basedir, 'static', 'data', 'tmp_audio', '*')):
        os.remove(fl)


def cleanup_data():
    """ Clean up temporary stored data
    """
    for fl in glob.glob(os.path.join(basedir, 'static', 'data', 'tmp_json', '*')):
        os.remove(fl)
