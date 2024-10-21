import os
import glob

experiments = glob.glob('./data/*')

for experiment in experiments:
    print(experiment)
    json = ['left', 'right']
    try:
        os.makedirs(experiment + '/json/' + json[0])
        os.makedirs(experiment + '/json/' + json[1])
    except OSError:
        pass
    for type in json:
        video = experiment + '/' + type + '_fitted_video.avi'
        write_json = experiment + '/json/' + type
        write_video = experiment + '/openpose_visualized_' + type + '.avi'
        os.system('./build/examples/openpose/openpose.bin --video {} --write_json {} --write_video {} --display 0'.format(video, write_json, write_video))