import math

import cv2
import cv2 as cv
import numpy as np

lk_params = dict(
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    winSize=(15, 15),
    maxLevel=2
)

feature_params = dict(
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)


class RodentTracker:
    def __init__(self, resize_height=480, max_flow=1000, max_features=50, feature_history=2, odometry_history=2, visualize=False):
        # resize frame height
        self.frame_height = resize_height

        # maximum allowed flow per frame
        self.max_flow = max_flow
        self.max_flow_squared = self.max_flow ** 2

        # feature parameters
        self.max_features = max_features
        self.feature_history = feature_history
        self.odometry_history = odometry_history

        self.visualize_frames = visualize

        self.frame_width = None

        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.prev_frame = None
        self.prev_gray = None
        self.frame = None
        self.visualization = None
        self.gray = None
        self.tile_map = None

        self.features = []
        self.path = np.array([[0, 0]])

    def preprocess(self):
        kernel = (5, 5)
        sig = 1

        split = cv.split(self.frame)
        self.frame = cv.merge([cv.GaussianBlur(split[x], kernel, sig) for x in (0, 1, 2)])

        if not self.frame_width:
            self.frame_width = math.ceil(self.frame.shape[1] / self.frame.shape[0] * self.frame_height)

        self.frame = cv.resize(self.frame, (self.frame_width, self.frame_height), cv.INTER_CUBIC)

    def enhance(self):
        parts = []

        bgr = cv.split(self.frame)
        bgr_clahe = cv.merge([self.clahe.apply(bgr[x]) for x in (0, 1, 2)])
        parts.append(bgr_clahe)

        lab = cv.cvtColor(self.frame, cv.COLOR_BGR2LAB)
        lab = cv.split(lab)
        lab_clahe = cv.merge([self.clahe.apply(lab[x]) for x in (0, 1, 2)])

        parts.append(cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR))

        hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        hsv = cv.split(hsv)
        hsv_clahe = cv.merge([self.clahe.apply(hsv[x]) for x in (0, 1, 2)])

        parts.append(cv.cvtColor(hsv_clahe, cv.COLOR_HSV2BGR))

        # Add parts using equal weighting
        if len(parts) > 0:
            weight = 1.0 / len(parts)
            blended = np.zeros(self.frame.shape)
            for p in parts:
                blended += weight * p
            self.frame = blended.astype(np.uint8)

        del blended

    def enhance_tiles(self):
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.gray = cv.GaussianBlur(gray, (7, 7), 0)

        self.tile_map = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    @staticmethod
    def mean_good_flow(flow):
        confidence_interval = 1.645
        flow = np.array(flow)

        # find flow vectors that are within some interval of the mean
        mean = np.mean(flow, axis=0)
        std_dev = np.std(flow, axis=0)
        conditions = np.abs(flow - mean) < std_dev * confidence_interval

        # mark "bad" and "good" flow vectors
        mask = (np.sum(conditions, axis=1) == 2).astype(np.int32)

        # return mean of "good" flow
        return np.mean(flow[mask == 1], axis=0)

    def read_frame(self, frame):
        # handle new frame read
        self.prev_frame = self.frame
        self.prev_gray = self.gray

        self.frame = frame

        # process pipeline
        self.preprocess()
        self.visualization = self.frame
        self.enhance()
        self.enhance_tiles()

    def run_optical_flow(self):
        # current features being tracked
        curr_features = np.float32([tr[-1] for tr in self.features]).reshape(-1, 1, 2)

        # find new positions of current features
        new_features, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, curr_features, None, **lk_params)

        # run optical flow in reverse to find old position of newly tracked features
        reverse_features, _st, _err = cv2.calcOpticalFlowPyrLK(self.gray, self.prev_gray, new_features, None,
                                                               **lk_params)

        # calculate distance between actual and computed previous feature positions
        # if they are too far off, then we discard those features as they are erroneous
        distance = abs(curr_features - reverse_features).reshape(-1, 2).max(-1)
        good_features = distance < 1

        new_tracks = []

        for tr, (x, y), good_feature in zip(self.features, new_features.reshape(-1, 2), good_features):
            if not good_feature:
                continue

            tr.append(np.array((x, y)))
            new_tracks.append(tr[-self.feature_history:])

        self.features = new_tracks

    def find_new_features(self):
        # ensure new features are forced to be present away from existing features
        for x, y in [np.int32(tr[-1]) for tr in self.features]:
            cv2.circle(self.tile_map, (x, y), 5, (0,), -1)

        # find required new features using Shi-Tomasi
        p = cv.goodFeaturesToTrack(
            self.gray,
            mask=self.tile_map,
            maxCorners=max(self.max_features - len(self.features), 0), **feature_params
        )

        # if there are any features detected
        # it is important that they are added to the front of the list
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.features.insert(0, [np.array((x, y))])

        # remove features that are too close to each other
        # newer features are prioritized over older features,
        # because they are earlier in the list

        i = 0
        while i < len(self.features):
            j = i + 1
            while j < len(self.features):
                if np.sum(np.square(self.features[i][-1] - self.features[j][-1])) < 50:
                    self.features.pop(j)
                else:
                    j += 1
            i += 1

    def compute_odometry(self):
        last_two_features = [tr[-2:] for tr in self.features if len(tr) >= 2]

        if len(last_two_features) >= 2:
            # Separate first elements (x,y tuples) and second elements (x,y tuples) into separate lists
            prev_features, curr_features = zip(*last_two_features)

            prev_features = np.array(prev_features)
            curr_features = np.array(curr_features)

            # compute flow vectors
            flow = curr_features - prev_features

            # filter out bad flow vectors, and get the mean of good flow
            good_flow = self.mean_good_flow(flow)

            # ensure good flow is not all zeros and is within a certain threshold
            if np.all(good_flow) and np.sum(np.square(good_flow)) < self.max_flow_squared:
                self.path = np.append(self.path - good_flow, [[0, 0]], axis=0)
                self.path = self.path[-self.odometry_history:]

    def visualize(self):
        cv2.polylines(self.visualization, [np.int32(tr) for tr in self.features], False, (0, 255, 0))

        if len(self.path) >= 2:
            cv2.polylines(self.visualization,
                          [np.array([self.frame_width // 2, self.frame_height // 2]) - np.int32(self.path)], False,
                          (0, 0, 255))

        # for centre in [np.int32(tr[-1]) for tr in self.features]:
        #     cv2.circle(self.visualization, centre, 2, (0, 255, 0), -1)

        cv2.imshow("frame", self.visualization)
        cv2.waitKey(1)

    def run_frame(self):
        # compute positions of new features for new frame
        if self.features:
            self.run_optical_flow()

        # compute new features if necessary
        if len(self.features) < self.max_features:
            self.find_new_features()

        # compute final flow vector
        self.compute_odometry()

        # create visuals and display
        if self.visualize_frames:
            self.visualize()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    tracker = RodentTracker(cap, visualize=True, odometry_history=20)

    while True:
        _, frame = cap.read()
        tracker.read_frame(frame)
        tracker.run_frame()
