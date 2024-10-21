dir = '../data/2022-11-25_19-35-05_testing_carpets_JY_squat_2';

load(dir + "/keypoints/keypoints3D/WX.mat");
load(dir + "/keypoints/keypoints3D/WY.mat");
load(dir + "/keypoints/keypoints3D/WZ.mat");

writematrix(WX, dir + "/keypoints/keypoints3D/WX.csv");
writematrix(WY, dir + "/keypoints/keypoints3D/WY.csv");
writematrix(WZ, dir + "/keypoints/keypoints3D/WZ.csv");