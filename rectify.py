import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os



class DocumentAligner:
    def __init__(self, image_folder_path, out_dir, debug_dir=""):
        self.image_folder_path = image_folder_path
        self.out_dir = out_dir
        self.debug_dir = debug_dir
        self.debug = debug_dir != ""

        # Preprocessing parameters
        self.blur_ksize = (7,7) # Gaussian Blur
        self.bilateral_ksize = 9 # Bilateral Filter
        self.downscale_factor = 0.25 # Scaling down the image
        self.clean_kernel_size = 5 # Morphological opening kernel size

    def load_image(self, image_num):
        image_path = f'{self.image_folder_path}/input ({image_num}).jpg'
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image, image_gray
    
    def rescale(self, image, image_num):
        small = cv2.resize(image, None, fx=self.downscale_factor, fy=self.downscale_factor)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/small.jpg', small)

        return small
    
    def gaussian_blur(self, image, image_num):
        blurred = cv2.GaussianBlur(image.copy(), ksize=self.blur_ksize, sigmaX=-1)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/blurred.jpg', blurred)

        return blurred
    
    def bilateral_blur(self, image, image_num):
        blurred = cv2.bilateralFilter(image.copy(), self.bilateral_ksize, 75, 75)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/bilateral_blurred.jpg', blurred)

        return blurred
    
    def otsu_threshold(self, image, image_num):
        otsuThresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/otsuThresh.jpg', otsuThresh)

        return otsuThresh
    
    def clean_threshold(self, otsuThreshold, image_num):
        kernel = np.ones((self.clean_kernel_size, self.clean_kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(otsuThreshold, cv2.MORPH_OPEN, kernel)
        if self.debug:
            cv2.imwrite(f'{self.debug_dir}/{image_num}/cleaned_threshold.jpg', cleaned)
        return cleaned
    
    def get_document_contour(self, image, threshold, image_num):
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxContour = max(contours, key=cv2.contourArea)
        maxContour_full = (maxContour / self.downscale_factor).astype(np.int32)
        if self.debug:
            output = image.copy()
            cv2.drawContours(output, [maxContour_full], -1, (255, 0, 255), 2)
            cv2.imwrite(f'{self.debug_dir}/{image_num}/contours.jpg', output)


        return maxContour
    
    def get_hull(self, contour):
        return cv2.convexHull(contour)
    
    def get_approx_points(self, hull, image, image_num):
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
        return (points / self.downscale_factor).astype(np.int32)
    
    def order_points(self, image, image_num, points):
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
    
    # TODO: clean this up
    # homogeneous linear least squares solution for finding a homography
    def findHomographyLLSQ(self, ptsLeft, ptsRight):
        # create the A matrix, which is a 2n x 9 matrix, where n is the number of points
        A = np.repeat(np.array([[0]*9] * len(ptsLeft), np.float32), 2, axis=0)
        for i in range(len(ptsLeft)):
            # TODO: your code, populate the A matrix
            # hint: use the ptsLeft and ptsRight arrays to get the points, and use the i index to put the points in the correct row
            # e.g. xL,yL = ptsLeft[i] will give you the left point, and xR,yR = ptsRight[i] will give you the right point

            xL,yL = ptsLeft[i]
            xR,yR = ptsRight[i]

            # according to the guidance above, we need to populate the A matrix with the following two rows for each point:
            #  [-xL -yL -1 0 0 0 xR*xL xR*yL xR]
            #  [0 0 0 -xL -yL -1 yR*xL yR*yL yR]
            # the row index can be figured out using the i index
            A[2*i] = [-xL, -yL, -1, 0, 0, 0, xR*xL, xR*yL, xR]
            A[2*i+1] = [0, 0, 0, -xL, -yL, -1, yR*xL, yR*yL, yR]

        # take the SVD of A (`np.linalg.svd`) and pick the last row of Vt (last column of V)
        # TODO: your code here...
        u,z,vt = np.linalg.svd(A)
        # the last row of Vt (e.g. [-1]) is the solution to the homogeneous (Ax=0) linear least squares problem
        # reshape the vector back to a 3,3 matrix
        H = vt[-1].reshape(3,3)
        # normalize the last element to 1 (homogeneous coordinates) by dividing H by the last [2,2] element of H
        return H / H[2,2]
    
    def apply_homography(self, image, ordered_points):
        # use the findHomographyLLSQ function you wrote to find the homography matrix 
        # # using the pts1 and pts2 arrays 
        dst_points = np.array([
            [0, 0],        # top-left
            [425-1, 0],    # top-right
            [425-1, 550-1],# bottom-right
            [0, 550-1]     # bottom-left
        ], dtype=np.float32)
        H = self.findHomographyLLSQ(ordered_points, dst_points) 
        # get the result using warpAndBlendImagesHomography 
        # lstsqWarp = warpAndblendImagesHomography(im1wide, im2, H)
        warped = cv2.warpPerspective(image, H, (425, 550))

        return warped

    def run(self, image_num):
        if self.debug:
            os.makedirs(f'{self.debug_dir}/{image_num}', exist_ok=True)
        image_color, image_gray = self.load_image(image_num)

        # Preprocessing and Binarization
        small = self.rescale(image_gray, image_num)
        blurred = self.bilateral_blur(small, image_num)
        otsuThreshold = self.otsu_threshold(blurred, image_num)
        cleaned = self.clean_threshold(otsuThreshold, image_num)

        # Feature and Contour Extraction
        contour = self.get_document_contour(image_color, cleaned, image_num)

        # Corner Detection / Localization
        hull = self.get_hull(contour)
        approx_points = self.get_approx_points(hull, image_color, image_num)
        approx_full = self.scale_up_points(approx_points)
        ordered_points = self.order_points(image_color, image_num, approx_full)

        # Part 4: Geometric Rectification
        warped = self.apply_homography(image_color, ordered_points)

        # save the final image to the output directory
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(f'{self.out_dir}/{image_num}_rectified.jpg', warped)
        print(f'Processed image {image_num}')

# TODO: landscape?
# TODO: handle error thrown 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_folder_path',
        type=str,
        help="path to the folder containing images",
        default="synthetic_data"
    )
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
        rectifier.run(i)
    

