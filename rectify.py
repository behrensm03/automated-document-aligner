import cv2
import numpy as np
import argparse
import os

class DocumentAligner:
    def __init__(self, image_folder_path, out_dir, debug_dir=""):
        self.image_folder_path = image_folder_path
        self.out_dir = out_dir
        self.debug_dir = debug_dir
        self.debug = debug_dir != ""

        # Preprocessing parameters
        self.bilateral_ksize = 9 # Bilateral Filter kernel size
        self.downscale_factor = 0.25 # scale the image down
        self.kmeans_k = 3 # Number of clusters for k-means thresholding

    def load_image(self, image_num):
        # Allowing the input images to come in either "input (1).jpg" or "input1.jpg" format
        possible_image_paths = [
            f'{self.image_folder_path}/input ({image_num}).jpg',
            f'{self.image_folder_path}/input{image_num}.jpg',
        ]
        for image_path in possible_image_paths:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                return image, image_gray

        raise FileNotFoundError(f"Could not find image for number {image_num} in expected formats.")
    
    def rescale(self, image, image_num):
        # Rescale the image down to speed up processing and reduce noise
        small = cv2.resize(image, None, fx=self.downscale_factor, fy=self.downscale_factor)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/small.jpg', small)

        return small
    
    def bilateral_blur(self, image, image_num):
        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(image.copy(), self.bilateral_ksize, 75, 75)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/bilateral_blurred.jpg', blurred)

        return blurred
    
    def kmeans_threshold(self, img, image_num):
        # Return a binary image where the document is white and the background is black, 
        # using k-means clustering on pixel intensities.
        pixels = img.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, self.kmeans_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # keep only the brightest cluster, assume that is the document
        document_label = np.argmax(centers)
        thresh = (labels.reshape(img.shape) == document_label).astype(np.uint8) * 255
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/kmeans_thresh.jpg', thresh)

        return thresh
    
    def get_document_contour(self, image, threshold, image_num):
        # Find contours in the thresholded image, and keep only the largest one, assuming it is the document.
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxContour = max(contours, key=cv2.contourArea)
        if self.debug:
            maxContour_full = (maxContour / self.downscale_factor).astype(np.int32)
            output = image.copy()
            cv2.drawContours(output, [maxContour_full], -1, (255, 0, 255), 2)
            cv2.imwrite(f'{self.debug_dir}/{image_num}/contours.jpg', output)

        return maxContour
    
    def get_hull(self, contour):
        # perform convex hull to smooth out the contour and get rid of any small imperfections
        return cv2.convexHull(contour)
    
    def get_approx_points(self, hull, image, image_num):
        # Find a set of 4 points that approximate the hull. 
        for scale in [0.02, 0.03, 0.04, 0.05, 0.06, 0.1]:
            epsilon = scale * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                if self.debug:
                    output = image.copy()
                    scaled = (approx / self.downscale_factor).astype(np.int32)
                    cv2.drawContours(output, [scaled], -1, (0, 255, 0), 2)
                    cv2.imwrite(f'{self.debug_dir}/{image_num}/hull.jpg', output)
                return approx
        
        raise ValueError(f"Could not find 4 points for image {image_num}")
    
    def scale_up_points(self, points):
        # Revert the corner points back to the original image scale
        return (points / self.downscale_factor).astype(np.int32)
    
    def order_points(self, image, image_num, points):
        # Deterministically order the corner points in the clockwise order: top-left, top-right, bottom-right, bottom-left
        sums = points.sum(axis=2)
        diffs = np.diff(points, axis=2)
        top_left = points[np.argmin(sums)]
        bottom_right = points[np.argmax(sums)]
        top_right = points[np.argmin(diffs)]
        bottom_left = points[np.argmax(diffs)]

        ordered = np.array([top_left, top_right, bottom_right, bottom_left])
        ordered = ordered.reshape(4, 2)

        if self.debug:
            output = image.copy()
            colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]  # TL, TR, BR, BL
            labels = ['TL', 'TR', 'BR', 'BL']
            for i, (x, y) in enumerate(ordered):
                cv2.circle(output, (x, y), 8, colors[i], -1)
                cv2.putText(output, labels[i], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colors[i], 2)
            cv2.imwrite(f'{self.debug_dir}/{image_num}/ordered_points.jpg', output)

        return ordered
    
    def apply_homography(self, image, ordered_points):
        # Warp the image using the homography defined by the detected corner points.
        dst_points = np.array([
            [0, 0],        # top-left
            [425-1, 0],    # top-right
            [425-1, 550-1],# bottom-right
            [0, 550-1]     # bottom-left
        ], dtype=np.float32)
        H, _ = cv2.findHomography(ordered_points, dst_points)
        warped = cv2.warpPerspective(image, H, (425, 550))

        return warped

    def run(self, image_num):
        if self.debug:
            # Ensure the debug directory for this image exists, if we are in debug mode
            os.makedirs(f'{self.debug_dir}/{image_num}', exist_ok=True)

        # Load the image
        image_color, image_gray = self.load_image(image_num)

        # Part 1: Preprocessing and Binarization
        small = self.rescale(image_gray, image_num)
        blurred_small = self.bilateral_blur(small, image_num)
        thresh = self.kmeans_threshold(blurred_small, image_num)

        # Part 2: Feature and Contour Extraction
        contour = self.get_document_contour(image_color, thresh, image_num)

        # Part 3: Corner Detection / Localization
        hull = self.get_hull(contour)
        approx_points = self.get_approx_points(hull, image_color, image_num)
        approx_full = self.scale_up_points(approx_points)
        ordered_points = self.order_points(image_color, image_num, approx_full)

        # Part 4: Geometric Rectification
        warped = self.apply_homography(image_color, ordered_points)

        # save the final image to the output directory
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(f'{self.out_dir}/output{image_num}.jpg', warped)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/final_warped.jpg', warped)
        print(f'Processed image {image_num}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required argument to point to input directory
    parser.add_argument(
        'image_folder_path',
        type=str,
        help="path to the folder containing images",
    )

    # Optional arguments for specifying where to write outputs, where to write debug info
    # or to only run a subset of the input images.
    parser.add_argument(
        '--out_dir',
        type=str,
        help="directory to save rectified images",
        default="output"
    )
    parser.add_argument(
        '--debug_dir',
        type=str,
        help="directory to save debug images",
        default=""
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        help="starting index of images to process",
        default=1
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        help="ending index of images to process",
        default=72
    )

    args = parser.parse_args()
    
    rectifier = DocumentAligner(args.image_folder_path, args.out_dir, args.debug_dir)
    for i in range(args.start_idx, args.end_idx + 1):
        try:
            rectifier.run(i)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    

