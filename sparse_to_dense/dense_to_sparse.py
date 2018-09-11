# coding=utf-8
import numpy as np
import cv2


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass


class UniformSampling(DenseToSparse):
    name = "uar"

    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%.4f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
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
                return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
        else:
            prob = float(self.num_samples) / depth.size
            return np.random.uniform(0, 1, depth.shape) < prob


class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=0, dilate_iterations=0):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

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
    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)

        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        mag = cv2.magnitude(gx, gy)
        non_zeros = np.count_nonzero(depth_mask)

        if non_zeros > 0:
            target_samples = 1.0 * self.num_samples / np.size(depth) * non_zeros
            edge_percentage = float(100 * target_samples) / np.size(depth)
            min_mag_lower = np.percentile(mag[depth_mask], 100 - edge_percentage, interpolation='lower')
            min_mag_upper = np.percentile(mag[depth_mask], 100 - edge_percentage, interpolation='higher')
            # print("Upper threshold %f with %d pixels above it" %
            #      (min_mag_upper, np.count_nonzero(np.bitwise_and(depth_mask, mag >= min_mag_upper))))
            # print("Lower threshold %f with %d pixels above it" %
            #      (min_mag_lower, np.count_nonzero(np.bitwise_and(depth_mask, mag >= min_mag_lower))))
            mag_mask = np.bitwise_and(mag >= min_mag_upper, depth_mask)

            missing_samples = target_samples - np.count_nonzero(mag_mask)
            # assert missing_samples >= 0
            #if missing_samples > 0:
                # print("Missing %d samples" % missing_samples)
                # If we take all of mag_fill_mask we have more than desired number of samples
            #    mag_fill_mask = np.bitwise_and(depth_mask, mag < min_mag_upper, mag >= min_mag_lower)
            #    idx_0, idx_1 = np.nonzero(mag_fill_mask)
            #    potential_samples = len(idx_0)
                # print("Got %d potential samples" % potential_samples)
            #    if potential_samples > missing_samples:
                    # We choose some random samples to fill in
            #        index_of_indices = np.random.choice(len(idx_0), missing_samples, replace=False)
                    # Set these randomly chosen potentials to 1
            #        mag_mask[idx_0[index_of_indices], idx_1[index_of_indices]] = 1
                    # The threshold is still the lowest possible
            #    else:
            #        mag_mask[idx_0, idx_1] = 1
                    # print("filling up only partly possible, as all pixels with depth <= max_depth are already taken")
            #        assert np.all(mag_mask[depth_mask] == 1)

            if self.dilate_iterations >= 0:
                kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
                cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)

            return mag_mask
        else:
            return depth_mask


# Pixels are sampled with probability ~ w_t * t + w_d / d²
# num_samples pixels are 'chosen without putting back' (ziehen ohne zurücklegen)
class SimulatedKinect(DenseToSparse):
    name = "sim_kinect"

    def __init__(self, num_samples, weight_magnitude=1.0, weight_depth=1.0):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.weight_magnitude = weight_magnitude
        self.weight_depth = weight_depth

    def __repr__(self):
        return "%s{ns=%d,wm=%.4f,wd=%.4f}" % \
               (self.name, self.num_samples, self.weight_magnitude, self.weight_depth)

    def dense_to_sparse(self, rgb, depth):
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
            indices = np.random.choice(np.size(depth), size=self.num_samples, replace=False, p=probs.flatten())
            assert len(indices) == len(set(indices))

            mask = np.zeros(np.size(depth), dtype=np.bool)
            mask[indices] = 1

            return np.reshape(mask, np.shape(depth))


class Contours(DenseToSparse):
    name = "contours"

    def __init__(self, num_samples, min_val=100, max_val=200):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return "%s{ns=%d,wm=%.4f,wd=%.4f}" % \
               (self.name, self.num_samples, self.weight_magnitude, self.weight_depth)

    def dense_to_sparse(self, rgb, depth):
        depth_mask = depth != 0.0

        non_zeros = np.count_nonzero(depth_mask)

        if non_zeros <= self.num_samples:
            return depth_mask
        else:
            gray = rgb2grayscale(rgb)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

            mag = cv2.magnitude(gx, gy)

            target_samples = 1.0 * self.num_samples / np.size(depth) * non_zeros
            edge_percentage = float(100 * target_samples) / np.size(depth)
            min_mag_upper = np.percentile(mag[depth_mask], 100 - edge_percentage, interpolation='higher')

            mag_mask = np.zeros(depth_mask.shape, np.uint8)
            mag_mask[np.bitwise_and(mag >= min_mag_upper, depth_mask)] = 255

            im2, contours, hierarchy = cv2.findContours(mag_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            chosen = np.random.choice([0, 1], size=len(contours), replace=True)

            print(chosen)

            mask = np.zeros(depth_mask.shape, np.uint8)
            cv2.drawContours(mask, [c for i, c in enumerate(contours) if chosen[i]], -1, 255, -1)

            return np.bitwise_and(depth_mask, mask != 0)

