import copy
import numpy as np
import math
from skimage import measure
import cv2
from sklearn.linear_model import LinearRegression

def get_slope_bias(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection_line(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def get_foot_direction(line1, line2):
    foot1 = line1
    foot2 = line2

    # get all possible angles
    foot1_v11 = (foot1[0] - foot1[1], foot1[0], foot1[1])
    foot1_v12 = (foot1[1] - foot1[0], foot1[1], foot1[0])
    foot2_v21 = (foot2[0] - foot2[1], foot2[0], foot2[1])
    foot2_v22 = (foot2[1] - foot2[0], foot2[1], foot2[0])

    vectors = [
        [foot1_v11, foot2_v21],
        [foot1_v11, foot2_v22],
        [foot1_v12, foot2_v21],
        [foot1_v12, foot2_v22]
    ]

    angles = []
    for vector1, vector2 in vectors:
        v1, _, _ = vector1
        v2, _, _ = vector2

        angle = vector_angle(v1, v2)
        angles.append([angle, vector1, vector2])
    angles.sort(key=lambda x: x[0])

    # check orthodromic or opposite
    corrct_angles = []
    for angle, vector1, vector2 in angles:
        v1, end1, start1 = vector1
        v2, end2, start2 = vector2

        point = intersection_line(get_slope_bias(start1, end1), get_slope_bias(start2, end2))

        alpha1 = ((point - start1)[v1 != 0]/v1[v1 != 0]).sum()
        alpha2 = ((point - start2)[v2 != 0]/v2[v2 != 0]).sum()

        v1_reachable = alpha1 > 0
        v2_reachable = alpha2 > 0

        if v1_reachable == v2_reachable:
            corrct_angles.append([angle, vector1, vector2])
    #assert len(corrct_angles) == 2

    # check front or back
    foot_vectors = []
    for _, vector1, vector2 in corrct_angles:
        v1, end1, start1 = vector1
        v2, end2, start2 = vector2

        start_dist = ((start1-start2)**2).sum()
        end_dist = ((end1 - end2) ** 2).sum()
        foot_vectors.append([vector1, vector2, end_dist - start_dist])

    foot_vectors.sort(key = lambda x:x[2], reverse=True)
    vector1, vector2, _ = foot_vectors[0]
    v1, end1, start1 = vector1
    v2, end2, start2 = vector2
    direction = (v1 + v2)/2
    return direction

def drop_small_chunks(image_, threshold=8):
    image = copy.deepcopy(image_)
    image[image!=0] = 1
    label_image = measure.label(image, connectivity=2)

    new_image = np.zeros_like(image)
    for i in range(1, label_image.max() + 1):
        idxs = np.where(label_image == i)
        if len(idxs[0]) > threshold:
            new_image[idxs] = image_[idxs]
    return new_image

def drop_noise(image):
    image -= image.min()
    threshold = image.mean() + image.std()
    filtered_image = copy.deepcopy(image)
    filtered_image[image < threshold] = 0
    return filtered_image

def get_linear_reg(x, y):
    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)
    error = ((pred-y)**2).mean()
    return model, error


def get_line(x, y):
    x_idxs = np.expand_dims(x, axis=-1)
    y_idxs = np.expand_dims(y, axis=-1)

    model1, error1 = get_linear_reg(x_idxs, y_idxs)
    model2, error2 = get_linear_reg(y_idxs, x_idxs)
    if error1 > error2:
        Y_minmax = np.array([[y_idxs.min()], [y_idxs.max()]])
        pred = model2.predict(Y_minmax)
        y_re = Y_minmax[:, 0].astype(np.int32).tolist()
        x_re = pred[:, 0].astype(np.int32).tolist()
        return np.array(list(zip(x_re, y_re)))
    else:
        X_minmax = np.array([[x_idxs.min()], [x_idxs.max()]])
        pred = model1.predict(X_minmax)
        x_re = X_minmax[:, 0].astype(np.int32).tolist()
        y_re = pred[:, 0].astype(np.int32).tolist()
        return np.array(list(zip(x_re, y_re)))


def blur_image(image):
    k = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]) * (1 / 9)
    image = cv2.filter2D(image, -1, k)
    return image

def get_chunks(image_, denoise=1, blur=1):
    image = copy.deepcopy(image_)

    if denoise > 0:
        for _ in range(denoise):
            image = drop_noise(image)

    if blur > 0:
        for _ in range(blur):
            image = blur_image(image)

    image[image != 0] = 1
    label_image = measure.label(image, connectivity=2)

    # get chunks
    chunks = []
    for i in range(1, label_image.max() + 1):
        idxs = label_image == i
        value = image[idxs].sum()
        chunks.append((i, idxs, value))
    chunks.sort(key=lambda x: x[2], reverse=True)
    return chunks

def unit_vector(v):
    return v/(v[0]**2+v[1]**2)**0.5

def vector_angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    inner_product /= (len1*len2)
    inner_product *= 0.98
    angle = math.acos(inner_product)
    return angle


class FootDetector:
    def __init__(self, visualize):
        self.visualize = visualize

    def __call__(self, images, **kwargs):
        '''
        INPUT
        images : (time, 64, 64)

        OUTPUT
        available: True if result angle and speed are available
        angle: angle of human
        speed: speed of human
        '''
        cumulative_activation = np.zeros_like(images[0])
        for image in images:
            # get chunks
            chunks = get_chunks(image)
            chunks = chunks[:2]
            # get idxs of activated area
            for _, idxs, _ in chunks:
                int_idxs = np.where(idxs)
                cumulative_activation[int_idxs] += image[int_idxs]

        # detect foots
        total_chunks = get_chunks(cumulative_activation, denoise=1, blur=1)

        # denoise and get direction
        available = True
        if len(total_chunks) > 1:
            two_chunks = total_chunks[:2]
            foot_infos = []
            viz = np.zeros_like(cumulative_activation)
            for _, idxs, _ in two_chunks:
                int_idxs = np.where(idxs)
                chunk = cumulative_activation[idxs]
                x_idxs = int_idxs[0].astype(int)
                y_idxs = int_idxs[1].astype(int)

                x_idxs = x_idxs[chunk != 0]
                y_idxs = y_idxs[chunk != 0]

                # upsampling idx for linear reg

                maxv, minv = 10, 1
                level_chunk = cumulative_activation[(x_idxs, y_idxs)]
                level_chunk = (level_chunk - level_chunk.min()) / (level_chunk.max() - level_chunk.min() + 1e-6)
                level_chunk = level_chunk*(maxv-minv) + minv
                level_chunk = level_chunk.astype(np.int32)
                total_idxs = []
                level_idxs = list(zip(level_chunk, x_idxs, y_idxs))
                for level, x, y in level_idxs:
                    total_idxs += [(x, y) for _ in range(level)]
                total_idxs = np.array(total_idxs)
                x_idxs, y_idxs = total_idxs[:, 0], total_idxs[:, 1]

                line = get_line(x_idxs, y_idxs)
                foot_infos.append((x_idxs, y_idxs, line))

                viz[idxs] = chunk
            # produce angle
            foot1 = foot_infos[0][-1]
            foot2 = foot_infos[1][-1]
            direction = get_foot_direction(foot1, foot2)
            global_center = (foot1[0] + foot1[1] + foot2[0] + foot2[1]) / 4
            direction_line = np.array([global_center, global_center + direction]).astype(np.int32)

            angle = math.degrees(math.atan2(direction[1], direction[0]))
            angle = angle + 180
            if angle > 180:
                angle = angle%180

            # produce speed
            diffs = []
            for _, idxs, _ in two_chunks:
                area = images[:, idxs]
                diff = abs(area[:-1] - area[1:])
                diff = diff.sum(axis=1)
                diffs.append(diff)
            speed = diffs[0].mean() + diffs[1].mean()

            if self.visualize:
                # visualize
                draw_lines = [
                    [foot1, (255, 0, 0)],
                    [foot2, (255, 0, 0)],
                    [direction_line, (0, 255, 0)],
                ]
                viz = images[-1]
                viz = (viz - viz.mean())/viz.std()
                viz = (viz+15)/30

                viz = (viz*255).astype(np.uint8)
                viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
                for line, color in draw_lines:
                    cv2.line(viz, line[0][::-1], line[1][::-1], color, 1)
                self.visualized_image = viz
        else:
            available, angle, speed = False, None, None
        return available, angle, speed
