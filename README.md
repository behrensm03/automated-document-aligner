## Pipeline Description

The pipeline contains four main stages: binarization, contour finding, corner localization, and geometric rectification. The binarization stage preprocesses the image with blurring and thresholding to create a clean binary image. The second stage utilizes the binary image to find contours in the image. The corner localization phase utilizes a convex hull, and an approximate polygon to find polygons that fit the contour with four corners. Finally, these four corners are used to compute a homography which warps the image to rectify the document.

## Binarization

- The binarization step begins with a downscaling of the (grayscale) image by a factor of 0.25, and then uses a bilateral filter to blur the image with a kernel size of 9. 
- The downscaling serves two purposes: it aids in processing speed, but more importantly, it helps to smooth over fine details that aren't strictly necessary. There can be detailed background textures that cause confusion with the thresholding step, but by downscaling, we can blur these minor details together into an easier texture so that we better remove this background noise. 
- The blurring step utilizes a bilateral filter, as opposed to a Gaussian filter, because the bilateral filter better preserves edges. The Gaussian filter smooths over the edges just as it does with other parts of the image, and will make those edges weaker. The bilateral filter allows the edges to remain strong which makes it easier to fit our contour and eventually find our corner points. 
- Finally, the downscaled and blurred image goes through a k-means process to convert the image into a binary image. This method of thresholding was chosen due to observed difficulties with Otsu and adaptive thresholding. In certain images in the dataset, there are very bright background patches immediately next to the document, and the Otsu and adaptive thresholding methods had difficulty segmenting these sections from the document itself without removing other parts of the document. Upon manual inspection, many of the problem images had bright patches in the background that were being treated as part of the main document contour. By segmenting using a k-means approach, the thresholded image came out much cleaner as it was able to differentiate between the brightness of the document and the blobs in the background.
#### Parameters:
- Downscale Factor: 0.25. I chose this as it provided a good balance between leaving enough detail to work with, but removing background noise.
- Bilateral Filter kernel size: 9. I experimented with larger and smaller filters and found that larger kernels blurred too strongly and made it difficult to fit contours accurately, while smaller kernels kept too much background detail which can confuse the contour process.
- K-Means: k=3. The idea here is that our problem images suffered from the same issue, which is that there were bright background blobs that were being treated as the document with Otsu and adaptive thresholding methods. By using k-means with k=3, it allows the possibility of a third bucket to exist. Instead of labeling every pixel as "bright" or "dark", it allows a middle ground of "medium brightness". This works well to resolve those problematic images because the background patches are bright but not quite as bright as the document. Inspiration for this method was taken from a class piazza post mentioning k-means as a possible strategy, and I realized it could solve the problem I was facing.

## Edge / Contour Finding

- The contour finding step of the pipeline is relatively straightforward. I did not use an edge detector, such as Canny, because I was able to preprocess the images sufficiently in the binarization phase that a simple contour combined with my corner localization step was producing good results. To find contours, I utilize `cv2.findContours`, and then assume that the largest contour found will be the document contour. Thus, I keep the largest contour by area and discard the rest, treating that large contour as the document.

#### Parameters
None

## Corner Localization

- The corner localization phase begins by fitting a convex hull to the document contour, using `cv2.convexHull`. The hull has the effect of turning a possibly jagged or concave contour into a simpler, convex shape. This helps remove noise created from a background patch, or from the text on the document itself representing a border of the contour. 
- Then, to find the corner points, we start with the hull and fit a polygon to that hull. This simplifies the hull into a polygon with exactly four vertices, using the Ramer-Douglas-Peucker algorithm via `cv2.approxPolyDP`. 
- Finally, the four vertices produced are rescaled back up to the original image scale, and ordered clockwise, starting from the top left. The order is determined using the sums and differences of the detected corner points. The top left is the point with the smallest x+y coordinate (closest to the origin in the top left), and the bottom right has the highest x+y. The top right has the smallest y - x difference in coordinates (large x coordinate, small y coordinate) and the bottom left has the largest difference (large y coordinate, small x coordinate).

#### Parameters
- `cv2.approxPolyDP` takes an epsilon parameter. This stage uses a trial and error approach, finding an epsilon that produces exactly four vertices. The possible epsilon values, which is a constant list that does not change per image, is [0.02, 0.03, 0.04, 0.05, 0.06, 0.1]. Whichever value first produces a polygon with 4 vertices is used.

## Geometric Rectification

- The final step is geometric rectification, which is relatively straightforward once the corner point locations are determined. We know the final image size will be 550x425 pixels, so we can define our destination points of the homography from this. And the starting points are the (ordered) corner points from the prior step. Then, I pass this into `cv2.findHomography` to compute the homography between these two sets of points, and finally call `cv2.warpPerspective` to apply that homography. The result is the final image.

#### Parameters
None

## Failure Modes / Observations

- One fundamental issue that will be readily visible from my selected sample output images, is that the process cannot handle situations where one of the document edges is curved or folded (see `output6.jpg` in the selected sample outputs). This is a fundamental limitation in our use of the homography to warp the perspective, as it assumes a planar surface and connects the detected corners with straight lines. Thus, even if the corners are accurate, if the document edge is not a straight line due to folding (and the edge is above or below the straight line), the straight line connecting those corners will either cut off part of the document edge or include some of the background, depending on which direction the curve is. This is not a failure of the pipeline, but a fundamental shortcoming of using a homography to warp the perspective.
- One other image, which is available in my output samples (see `output46.jpg`), has an issue where the detected top right corner is actually inside the document. The reason for this issue is that the binary image produced by the k-means threshold leaves a small gap between the true top right corner and the rest of the document, so the small section of the document is excluded from the main contour. This could be because that portion of the document has a shadow or is lower in brightness than the rest of the document and therefore is treated as part of the second cluster instead of the brightest.
- Other images may contain small amounts of noise in their corner placement, especially when the background is particularly bright close to the document, or where the document itself contains darker regions or shadows. This is one of the limitations of using a brightness-based approach to perform the segmentation of the document from its background.

## Selected Sample Images in `output_samples`

- `output6.jpg`: this showcases an example of the limitation of the homography method of connecting corners with straight lines. Because the document is warped, the top edge has two curves: on the right half of the document, the top edge curves downwards, and on the left half it curves upwards. The result is that even though the corner detection is accurate, connecting the corners with a straight line means we slightly cut off the top on the left half, and include some of the background on the right half.
- `output15.jpg`: this showcases a relatively clean rectification of the document, because the original document photo has little warping and thus can be approximated well with a polygon. This also was a particulary difficult background to work with, as there is a bright background patch right next to the document. Only the k-means thresholding was able to produce a clean enough segmentation to get a rectification of this quality.
- `output22.jpg`: this showcases a relatively clean rectification, in a case with a darker background.
- `output46.jpg`: this is one of the failure modes discussed above, where the detected corner point is actually inside the true corner (top right). This may be due to lower brightness in that part of the document. As a further visualization, `output46_polygon.jpg` shows the approximated polygon we fit and use to conduct the morph. It is clear in this image that we are cutting off part of the document.
- `output67.jpg`: another relatively clean example, with a dark background but a bright patch in the background that is again connected to the document.

### Important notes on rectify.py
- The script assumes images are named in the format `input (N).jpg` in the input directory provided, though the writeup document specifies it should look for `inputN.jpg`. This is because the actual dataset uses the former naming convention. If the file doesn't exist, it falls back to `inputN.jpg` format. If the script must be run on images with a different naming convention, a one-line edit will need to be made to the script in the `load_image` function.
- The script also has `--start_idx` and `--end_idx` parameters which are defaulting to 1 and 72, which is the size of the provided dataset. If the script needs to be run on a different dataset, you can pass in the appropriate values for these arguments. They define what indices to actually run on.
- If you supply a value to the `--debug_dir` directory, you will also get a set of intermediate images from the pipeline, showing the progress at various checkpoints (binarization, contour, corner detection, etc). This is optional and can be omitted.
