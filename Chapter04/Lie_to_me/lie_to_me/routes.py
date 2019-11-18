""" Routes of Flask Web Application """
import os
import csv
import threading
import shelve
from pathlib import Path
from flask import render_template, request, jsonify, url_for
from lie_to_me import app, video, basedir
from lie_to_me.process import process_video, process_audio, cleanup_uploads, predict

video_file_name = [None]  # Save video filename for later retrieval
prediction = [None]  # Save prediction results to display on results endpoint


@app.route('/', methods=['GET', 'POST'])
def upload():
    """ Home Page Route
        Upload and Analyze Video
    :return: home.html template
    """
    global video_file_name

    if request.method == 'POST' and 'file' in request.files:

        # clear uploads folder
        cleanup_uploads()

        # Video has been uploaded
        filename = video.save(request.files['file'])
        video_file_name[0] = filename

        # Process video on a new thread
        # threading.Thread(target=process_video, args=[os.path.join('uploads', filename)]).start()
        # threading.Thread(target=process_audio, args=[os.path.join('uploads', filename)]).start()
        threading.Thread(target=process_video, args=[os.path.join(basedir, 'static', 'data', 'uploads', filename)]).start()
        threading.Thread(target=process_audio, args=[os.path.join(basedir, 'static', 'data', 'uploads', filename)]).start()

        return jsonify({'success': True}), 200, {'ContentType': 'application/json'}

    elif request.method == 'GET':
        return render_template('home.html')


@app.route('/analysis', methods=['GET'])
def analysis():
    """ Analysis Route
        View Final Analysis of Uploaded Video
    :return:
    """
    global prediction

    json_path = os.path.join(basedir, 'static', 'data', 'tmp_json')
    # csv_path = os.path.join(basedir, 'static', 'data', 'csv')
    # if not os.path.exists(csv_path):
    #    os.mkdir(csv_path)

    if os.name == 'nt':
        audio_file = Path(os.path.join(json_path, 'audio_data.shlf.dir'))
        video_file = Path(os.path.join(json_path, 'facial_data.shlf.dir'))
    else:
        audio_file = Path(os.path.join(json_path, 'audio_data.shlf'))
        video_file = Path(os.path.join(json_path, 'facial_data.shlf'))

    # Files exists
    if audio_file.is_file() and video_file.is_file():
        with shelve.open(os.path.join(json_path, 'facial_data.shlf')) as shelf:
            emotion_data = shelf['emotion_data']
            microexpression_data = shelf['micro_expression_data']
            blink_data = shelf['blink_data']

        with shelve.open(os.path.join(json_path, 'audio_data.shlf')) as shelf:
            mean_energy = shelf['mean_energy']
            max_pitch_amp = shelf['max_pitch_amp']
            vowel_duration = shelf['vowel_duration']
            pitch_contour = shelf['pitch_contour']

    else:
        emotion_data = None
        microexpression_data = None
        blink_data = None
        mean_energy = None
        max_pitch_amp = None
        vowel_duration = None
        pitch_contour = None

    # Training Files (choose one)
    # soc_file = os.path.join(basedir, 'static', 'data', 'train_files', 'first_soc.txt')
    # niko_file = os.path.join(basedir, 'static', 'data', 'train_files', 'first_niko.txt')
    # vero_file = os.path.join(basedir, 'static', 'data', 'train_files', 'first_vero.txt')

    # txt_file = soc_file

    # train_data = []

    # for cases where one parameter has more elements
    # for i in range(min(len(blink_data), len(microexpression_data), len(mean_energy))):
    #    train_data.append(0)

    # train_file = open(txt_file)

    # for line in train_file:
    #    index1 = int((int(line[4]) * 600) + ((int(line[5]) * 60) + (int(line[7]) * 10) + int(line[8])) / 2)
    #    index2 = int((int(line[10]) * 600) + ((int(line[11]) * 60) + (int(line[13]) * 10) + int(line[14])) / 2)
    #    if line[0] == 'F':
    #        train_data[index1] = 1
    #        train_data[index2] = 1

    # with open(os.path.join(csv_path, 'train.csv'), 'w', newline='') as csv_file:
    #    writer = csv.writer(csv_file)
    #    writer.writerow(['Time Interval', 'Micro-expressions', 'Blinks',
    #                     'Mean Energy', 'Max Pitch Amplitude', 'Vowel Duration', 'Fundamental Frequency',
    #                     'False/True'])

    #    # for cases where one parameter has more elements than another
    #    for index in range(min(len(mean_energy), len(blink_data), len(microexpression_data))):
    #        writer.writerow([index, microexpression_data[index], blink_data[index],
    #                         mean_energy[index], max_pitch_amp[index], vowel_duration[index], pitch_contour[index],
    #                         train_data[index]])

    # finalresults = [['Time Interval', 'Micro-expressions', 'Blinks',
    #                 'Mean Energy', 'Max Pitch Amplitude', 'Vowel Duration', 'Fundamental Frequency' ]]
    final_results = []

    for index in range((min(len(mean_energy), len(blink_data), len(microexpression_data)))):
        final_results.append([microexpression_data[index], blink_data[index],
                             mean_energy[index], max_pitch_amp[index], vowel_duration[index],
                             pitch_contour[index]])

    prediction[0] = predict(final_results)

    return render_template('analysis.html', mean_energy=mean_energy, max_pitch_amp=max_pitch_amp,
                           vowel_duration=vowel_duration, pitch_contour=pitch_contour, blink_data=blink_data,
                           microexpression_data=microexpression_data, emotion_data=emotion_data)


@app.route('/results', methods=['GET'])
def results():
    """ Results Route
        View Final Lie Detection results of Uploaded Video
    :return:
    """
    # Create Mock truth/lie data
    from random import randrange
    # lie_results = [randrange(0, 2) for i in range(0, 50)]
    lie_results = [ int(i) for i in prediction[0] ]

    # Find lie timestamps
    # divmod to find minute, seconds -> results should be (min, sec, min, sec)
    # the first min,sec is start time of interval and second is end time
    lie_timestamps = []
    for index, value in enumerate(lie_results):
        if value:
            # a lie occurred
            start_m, start_s = divmod(index*2, 60)
            end_m, end_s = divmod(index*2 + 2, 60)

            lie_timestamps.append(
                ('{:02d}:{:02d}'.format(start_m, start_s), '{:02d}:{:02d}'.format(end_m, end_s))
            )

    video_src = url_for('static', filename='data/uploads/{0}'.format(video_file_name[0]))
    return render_template('results.html', lie_results=lie_results, lie_timestamps=lie_timestamps,
                           video_src=video_src)
