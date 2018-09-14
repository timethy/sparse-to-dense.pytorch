#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

import cv2


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


class DenseToSparse:
    def __init__(self, apply_kinect_noise=True):
        self.apply_kinect_noise = apply_kinect_noise
        pass

    def dense_to_sparse(self, rgb, depth, seed=None):
        if seed is not None:
            np.random.seed(seed)
        mask_keep = self.depth_mask(rgb, depth)
        sparse_depth = np.zeros(depth.shape, dtype=np.float32)

        if self.apply_kinect_noise:
            dm04 = depth[mask_keep] - 0.4
            sparse_depth[mask_keep] = np.random.normal(depth[mask_keep], 0.0019 * dm04 * dm04 + 0.0012)
        else:
            sparse_depth[mask_keep] = depth[mask_keep]

        return sparse_depth

    def depth_mask(self, rgb, depth):
        pass

    def __repr__(self):
        pass


class UniformSampling(DenseToSparse):
    name = "uar"

    def __init__(self, num_samples, max_depth=np.inf, apply_kinect_noise=True):
        DenseToSparse.__init__(self, apply_kinect_noise)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%.4f}" % (self.name, self.num_samples,
                                      self.max_depth)

    def depth_mask(self, rgb, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        if self.max_depth is not np.inf:
            mask_keep = depth <= self.max_depth
            n_keep = np.count_nonzero(mask_keep)
            if n_keep == 0:
                return mask_keep
            else:
                prob = float(self.num_samples) / n_keep
                return np.bitwise_and(
                    mask_keep,
                    np.random.uniform(0, 1, depth.shape) < prob)
        else:
            prob = float(self.num_samples) / depth.size
            return np.random.uniform(0, 1, depth.shape) < prob


class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"

    def __init__(self,
                 num_samples,
                 max_depth=np.inf,
                 dilate_kernel=0,
                 dilate_iterations=0,
                 apply_kinect_noise=True):
        DenseToSparse.__init__(self, apply_kinect_noise)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations
        self.kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)

    def __repr__(self):
        return "%s{ns=%d,md=%.4f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    # We do not use cv2.Canny, since that applies non max suppression
    # So we simply do
    # RGB to intensitities
    # Smooth with gaussian
    # Take simple sobel gradients
    # Threshold the edge gradient
    # Dilatate
    def depth_mask(self, rgb, depth):
        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

        non_zeros = np.count_nonzero(depth_mask)

        if non_zeros > 0:
            gray = rgb2grayscale(rgb)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

            mag = cv2.magnitude(gx, gy)
            target_samples = 1.0 * self.num_samples / np.size(depth) * non_zeros
            edge_percentage = float(100 * target_samples) / np.size(depth)
            min_mag_upper = np.percentile(
                mag[depth_mask], 100 - edge_percentage, interpolation='higher')
            mag_mask = np.bitwise_and(mag >= min_mag_upper, depth_mask)

            if self.dilate_iterations >= 0:
                return cv2.dilate(
                    mag_mask.astype(np.uint8),
                    self.kernel,
                    iterations=self.dilate_iterations) > 0

            return mag_mask
        else:
            return depth_mask


# Pixels are sampled with probability ~ w_t * t + w_d / d²
# num_samples pixels are 'chosen without putting back' (ziehen ohne zurücklegen)
class SimulatedKinect(DenseToSparse):
    name = "sim_kinect"

    def __init__(self,
                 num_samples,
                 weight_magnitude=1.0,
                 weight_depth=1.0,
                 apply_kinect_noise=True):
        DenseToSparse.__init__(self, apply_kinect_noise)
        self.num_samples = num_samples
        self.weight_magnitude = weight_magnitude
        self.weight_depth = weight_depth

    def __repr__(self):
        return "%s{ns=%d,wm=%.4f,wd=%.4f}" % \
               (self.name, self.num_samples, self.weight_magnitude, self.weight_depth)

    def depth_mask(self, rgb, depth):
        gray = rgb2grayscale(rgb)

        depth_mask = depth != 0.0
        non_zeros = np.count_nonzero(depth_mask)

        if non_zeros <= self.num_samples:
            return depth_mask
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

            mag = cv2.magnitude(gx, gy)

            # rescale depth to 0..1
            max_depth = np.max(depth)
            depth_ = depth / max_depth

            depth_sq = depth_ * depth_

            #print(np.max(mag))
            #print(np.max(1.0/depth_sq[depth_mask]))

            probs = self.weight_magnitude * mag / np.max(mag)
            #print(np.max(probs))
            probs[depth_mask] += self.weight_depth / depth_sq[depth_mask]
            #print(np.max(probs))
            probs[~depth_mask] = 0.0
            probs /= np.sum(probs)
            indices = np.random.choice(
                np.size(depth),
                size=self.num_samples,
                replace=False,
                p=probs.flatten())
            assert len(indices) == len(set(indices))

            mask = np.zeros(np.size(depth), dtype=np.bool)
            mask[indices] = 1

            return np.reshape(mask, np.shape(depth))


class Contours(DenseToSparse):
    name = "contours"

    def __init__(self,
                 probability,
                 edge_percentage=0.6,
                 apply_kinect_noise=True):
        DenseToSparse.__init__(self, apply_kinect_noise)
        self.probability = probability
        self.edge_percentage = edge_percentage

    def __repr__(self):
        return "%s{p=%.4f,ep=%.4f}" % (self.name, self.probability, self.edge_percentage)

    def depth_mask(self, rgb, depth):
        depth_mask = depth != 0.0

        gray = rgb2grayscale(rgb)

        # gray = depth
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        mag_rgb = cv2.magnitude(gx, gy)

        min_mag_upper_rgb = np.percentile(
            mag_rgb[depth_mask], self.edge_percentage, interpolation='higher')

        blurred = cv2.GaussianBlur(depth, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        mag_depth = cv2.magnitude(gx, gy)

        mag_mask = np.zeros(depth_mask.shape, np.uint8)

        min_mag_upper_depth = np.percentile(
            mag_depth[depth_mask], self.edge_percentage, interpolation='higher')

        mag_mask[mag_rgb >= min_mag_upper_rgb] = 255
        mag_mask[mag_depth >= min_mag_upper_depth] = 255

        # fig = plt.figure()
        # fig.add_subplot(3, 1, 1)
        # plt.imshow(mag_mask_rgb)
        # fig.add_subplot(3, 1, 2)
        # plt.imshow(mag_mask_depth)
        # fig.add_subplot(3, 1, 3)
        # plt.imshow(mag_mask)
        # plt.show()

        im2, contours, hierarchy = cv2.findContours(
            mag_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        avg_depth = np.zeros(len(contours))
        for i in range(0, len(contours)):
            valid_depth_points = 0.0
            for contour_point in contours[i]:
                depth_m = depth[contour_point[0][1], contour_point[0][0]]
                if depth_m > 0.0:
                    avg_depth[i] = avg_depth[i] + depth_m
                    valid_depth_points = valid_depth_points + 1.0
            if valid_depth_points > 0.0:
                avg_depth[i] = avg_depth[i] / valid_depth_points
            else:
                avg_depth[i] = 0.0

        # print "avg_depth: ", avg_depth
        # print "avg_depth_probabilities: ", avg_depth_probabilities
        # print "norm avg_depth_probabilities: ", np.linalg.norm(
        #     avg_depth_probabilities, ord=1)
        # print "probs: ", len(avg_depth_probabilities)
        # print "contours: ", len(contours)

        # Choose contours at random with a probability that depends on the
        # square of the distance to the camera.
        avg_depth_squared = np.square(avg_depth)
        contour_weights = self.probability * np.subtract(
            1.0, np.divide(avg_depth_squared, np.amax(avg_depth_squared)))

        chosen_random = np.random.random(len(contours)) < contour_weights

        #print("selected " + str(np.count_nonzero(chosen_random)) +
        #      " contours at random.")
        #print("selected " + str(np.count_nonzero(chosen_area)) +
        #      " contours because of small area.")

        #print("selected " + str(np.count_nonzero(chosen)) + " in total.")

        mask = np.zeros(depth_mask.shape, np.uint8)
        # Automatically select contours that have a low area, because
        # it is basically mostly noise or double edges from canny
        # edge detection.
        cv2.drawContours(
            mask, [c for i, c in enumerate(contours) if chosen_random[i] or cv2.contourArea(c) < 1000],
            thickness=-1,
            color=255,
            contourIdx=-1)

        return mask != 0


class Superpixels(DenseToSparse):
    name = "superpixels"

    def __init__(self, num_samples, apply_kinect_noise=True):
        DenseToSparse.__init__(self, apply_kinect_noise)
        self.num_samples = num_samples

    def __repr__(self):
        return "%s{ns=%d}" % \
               (self.name, self.num_samples)

    def depth_mask(self, rgb, depth):
        depth_mask = depth != 0.0

        print("plipp")

        num_superpixels = 10000

        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            np.size(rgb, 1),
            np.size(rgb, 0),
            3,
            num_levels=10,
            double_step=True,
            num_superpixels=num_superpixels)

        seeds.iterate(rgb, 10)

        labels_out = seeds.getLabels()

        debug_mask = seeds.getLabelContourMask(rgb, thick_line=True)
        plt.imshow(debug_mask)
        plt.show()

        if (True):
            im2, contours, hierarchy = cv2.findContours(
                debug_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            chosen_area = np.zeros(len(contours), dtype=bool)
            areas = np.zeros(len(contours))
            avg_depth = np.zeros(len(contours))
            for i in range(0, len(contours)):
                valid_depth_points = 0.0
                for contour_point in contours[i]:
                    depth_m = depth[contour_point[0][1], contour_point[0][0]]
                    if (depth_m > 0.0):
                        avg_depth[i] = avg_depth[i] + depth_m
                        valid_depth_points = valid_depth_points + 1.0
                if valid_depth_points > 0.0:
                    avg_depth[i] = avg_depth[i] / valid_depth_points
                else:
                    avg_depth[i] = 0.0
                areas[i] = cv2.contourArea(contours[i])

                # Automatically select contours that have a low area, because
                # it is basically mostly noise or double edges from canny
                # edge detection.
                chosen_area[i] = (areas[i] < 10)

            # print "avg_depth: ", avg_depth
            # print "avg_depth_probabilities: ", avg_depth_probabilities
            # print "norm avg_depth_probabilities: ", np.linalg.norm(
            #     avg_depth_probabilities, ord=1)
            # print "probs: ", len(avg_depth_probabilities)
            # print "contours: ", len(contours)

            # Choose contours at random with a probability that depends on the
            # square of the distance to the camera.
            avg_depth_squared = np.square(avg_depth)
            contour_weights = np.subtract(
                1.0, np.divide(avg_depth_squared, np.amax(avg_depth_squared)))

            for i in range(0, len(contours)):
                print("contour: " + str(i) + " avg depth: " +
                      str(avg_depth[i]) + " area: " + str(areas[i]) +
                      " weight: " + str(contour_weights[i]) + " choses: " +
                      str(chosen_area[i]))

            chosen_random = np.zeros(len(contours), dtype=bool)
            for i in range(0, len(contours)):
                chosen_random[i] = (np.random.random() < contour_weights[i])
                # print "contour ", i, " probability: ", contour_weights[
                #     i], " chosen: ", chosen_random[
                #         i], " distance: ", avg_depth[i]

            #
            # chosen_random = np.random.choice(
            #     range(0, len(contours)),
            #     size=len(contours),
            #     replace=False,
            #     p=avg_depth_probabilities)

            chosen = np.bitwise_or(chosen_random, chosen_area)

            print("selected " + str(np.count_nonzero(chosen_random)) +
                  " contours at random.")
            print("selected " + str(np.count_nonzero(chosen_area)) +
                  " contours because of small area.")

            print("selected " + str(np.count_nonzero(chosen)) + " in total.")

            mask = np.zeros(depth_mask.shape, np.uint8)
            cv2.drawContours(
                mask, [c for i, c in enumerate(contours) if chosen[i]],
                thickness=-1,
                color=255,
                contourIdx=-1)
            # cv2.drawContours(
            #     mask, [c for i, c in enumerate(contours) if chosen[i]],
            #     thickness=1,
            #     color=255,
            #     contourIdx=-1)

            return np.bitwise_and(depth_mask, mask != 0)
        else:
            randomness = "uar"
            # if randomness == "uar":
            choices = np.random.choice(
                [0, 1], size=num_superpixels, replace=True, p=[0.33, 0.67])
            chosen = [i for i, choice in enumerate(choices) if choice == 1]

            # else:
            #     # Thinking in process here...
            #     probs = np.ones(np.shape(depth_mask), np.uint32)
            #     chosen = []
            #     choice = np.random.choice(
            #         num_labels, size=1, replace=False, p=probs)
            #
            #     chosen_pixels = labels_out == choice
            #
            #     for i in range(self.num_samples):
            #         chosen += choice

            mask = np.zeros(depth_mask.shape, np.bool)

            for i in chosen:
                mask[labels_out == i] = 1

            return np.bitwise_and(depth_mask, mask)
