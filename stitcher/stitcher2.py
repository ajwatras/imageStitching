# import the necessary packages
import numpy as np
import imutils
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Variables used to change stitcher settings. ie. Tunable Parameters

# set to 1 to blend images (change to boolean?)
BLEND_IMAGES = 0
# Necessary?
EDGE_CORRECTION = 0
# Necessary?
DEBUGGING = 0
# Size of kernel dilation and erosion
DILATION_KERNEL = np.ones([3, 3])
# Number of times erosion function is performed
EROSION_LOOPS = 1
# Number of times dilation fucnction is performed
DILATION_LOOPS = 6
# Necessary?
EDGE_WIN_SIZE = 40

# What do the below variables do
SEAM_PAD = 45
# Number for detecting and discarding bad transformation (if image is too large, discard)
SIZE_BOUNDS = [5000, 5000]
# Necessary?
SEAM_ADJ = [-SEAM_PAD, SEAM_PAD, -SEAM_PAD, SEAM_PAD]


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    # Description: Main stitching function
    # Inputs: Explicit self, two images to be stitched, ratio for Lowe's test, noise threshold,
    # boolean for if matches should be showed, boolean for if the image is being restitched (unused)
    # Outputs: The final result as an image after the two inputs have been stitched together,
    # an image with the matchpoint visualized (optional), the homography matrix,
    # mask1 an array that shows the edge of image1, mask2 an array that shows the edge of image 2,
    # the coordinate shift as an array
    def stitch(self, images, ratio=.75, reprojThresh=4.0,
               showMatches=True, reStitching=False, seam=0):

        # unused as restitching code is never used
        if reStitching:
            print "Re stitching video"

        result = None
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        t = time.time()
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Necessary?
        # if seam is not 0:
        #    #In this case Seam should actually be MaskB or else we need a way to separate which points are outside the seam.
        #    print "Moving Anchor points to imageB"
        # Separate out the keypoints that are outside of imageB seam line.
        #    kpsA_temp = kpsA
        #    featuresA_temp = featuresA
        # Calculate distances of kpsA from the seam
        #    print "Calculating minimum distance"
        #    kpsA_distance = self.calcMinDistance(kpsA_temp,seam)
        # sort kpsA by closeness to imageA
        #    print "Sorting"
        #    Sorted_kpsA_idx = np.argsort(kpsA_distance)
        # grab set of features in kpsA which are farthest from seam line
        #    chosen_features = Sorted_kpsA_idx[-5:]
        # add those features to kpsB and featuresB
        #    for k in chosen_features:
        #       print "Adding feature ", k
        #        kpsB.append(kpsA_temp[k])
        #        featuresB.append(featuresA_temp[k])

        # prints time taken to detect keypoints
        elapsed = time.time() - t
        print "Detecting keypoints: %f Seconds" % elapsed

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            print 'Error: No Matching Features'

            if reStitching:
                return [0, 0, 0, 0, 0, (0, 0)]

            return [0, 0, 0, 0, 0, (0, 0)]

        # otherwise, apply a perspective warp to stitch the images
        # together

        t = time.time()

        (matches, H, status) = M
        # prints error if no homography is found
        if H is None:
            print "ERROR: no valid Homography"

            # Necessary? This if statement seems to be redundant
            if reStitching:
                return [0, 0, 0, 0, 0, (0, 0)]

            return [0, 0, 0, 0, 0, (0, 0)]

        # shifts the images to overlap
        result1, result2, coord_shift = self.shiftImage(imageA, imageB, H)
        if (result1 is 0) or (result2 is 0):
            if reStitching:
                # Necessary? This if statement seems to be redundant
                return [0, 0, 0, 0, 0, (0, 0)]

            return [0, 0, 0, 0, 0, (0, 0)]

        # Necessary?
        # Detect the appropriate size for the resulting image.
        # corners = np.array([[0,0,1],[0,imageA.shape[0],1],[imageA.shape[1],0,1],
        #	[imageA.shape[1],imageA.shape[0],1]]).T

        # print H, corners
        # img_bounds = np.dot(H,corners)

        # x_bound = np.divide(img_bounds[0,:],img_bounds[2,:])
        # y_bound = np.divide(img_bounds[1,:],img_bounds[2,:])
        # x_shift = 0
        # y_shift = 0
        # if min(x_bound) < 0:
        #	x_shift = -int(min(x_bound))
        # if min(y_bound) < 0:
        #	y_shift = -int(min(y_bound))
        # coord_shift = np.array([y_shift,x_shift])
        # shift_H = np.array([[1,0,x_shift],[0,1,y_shift],[0,0,1]])
        # x_bound = int(max(max(x_bound),imageB.shape[1]))
        # y_bound = int(max(max(y_bound),imageB.shape[0]))

        # print "X Bound:",x_bound,"Y Bound:",y_bound, "x shift:",x_shift,"y shift:",y_shift
        # if (x_bound+x_shift > SIZE_BOUNDS[0]) or (y_bound+y_shift > SIZE_BOUNDS[1]):
        #    print x_bound, y_bound
        #    print "ERROR: Image Too Large"
        #    if reStitching:
        #        return [0,0,0,0,0,(0,0)]
        #    return [0,0,0,0,0,(0,0)]

        # Warp Image A and place it in frame.
        # imageB2 = np.pad(imageB,((y_shift,0),(x_shift,0),(0,0)),'constant',constant_values = 0)
        # result2 = cv2.warpPerspective(imageA, np.dot(shift_H,H),
        #	(x_bound+x_shift,y_bound+y_shift))

        # result1 = np.pad(imageB2,((0,y_bound+y_shift - imageB2.shape[0]),(0,x_bound+x_shift - imageB2.shape[1]),(0,0)),'constant', constant_values=0)

        # Gradient blending (Move into blend)
        mask1 = (result1 > 0).astype('int')
        mask2 = (result2 > 0).astype('int')

        # blends images
        if BLEND_IMAGES == 1:

            mask = mask1 + mask2
            mask = mask + (mask == 0).astype('int')

            result = np.divide(result1.astype(int) + result2.astype(int), mask).astype('uint8')

        else:
            # print result1.shape
            mask = (result1 == 0).astype('int')
            result = (result2 * mask + result1).astype('uint8')

        # prints the time finding matches, shifting and blending took
        elapsed = time.time() - t
        print "Applying transformation: %f Seconds" % elapsed
        print "Within stitching coord shift", coord_shift

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            if reStitching:
                return (result, vis, H, mask1, mask2, coord_shift)

            return (result, vis, H, mask1, mask2, coord_shift)

        # return the stitched image

        return (result, H, mask1, mask2, coord_shift)

    # Description: Finds keypoints and features of an image
    # Inputs: explicit self, an image, mask (unused)
    # Outputs: keypoints as a tuple and features as a tuple
    def detectAndDescribe(self, image, mask=None):
        # convert the image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        # descriptor = cv2.xfeatures2d.SIFT_create() Necessary?
        descriptor = cv2.xfeatures2d.SURF_create()

        (kps, features) = descriptor.detectAndCompute(image, mask)

        # image_kps = image.copy()
        # cv2.drawKeypoints(image_kps,kps,image_kps)
        # cv2.imshow("KPS",image_kps)
        # cv2.waitKey(0)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # mask = (gray > 0).astype(float)
        # usable_points = np.nonzero(mask)
        # usable_points = zip(usable_points[0],usable_points[1])
        # print "Usable Points",usable_points
        # print "KeyPoints", kps
        # kps_final = []
        # features_final = []
        # for k in range(0,len(kps)):
        #    print np.around(kps[k])
        #    print zip(usable_points[0],usable_points[1])
        #    print tuple(np.around(kps[k])) in zip(usable_points[0],usable_points[1])
        #    if tuple(np.around(kps[k])) in usable_points:
        #        print kps[k],features[k]
        #        kps_final.append(kps[k])
        #        features_final.append(features[k])

        # kps = kps_final
        # features = features_final

        # print kps, features

        # return a tuple of keypoints and features
        return (kps, features)

    # Description: Finds match points and the homography matrix for two images
    # Inputs: Explicit self, tuple of keypoints for image1, tuple of keypoints for image 2,
    # ratio for Lowe's test, noise threshold
    # Outputs: An array of matches, the homography matrix, Status flag for if findHomography() completed successfully
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):

        # Compute the raw matches and initialize the list of actual matches

        # records time at starts of process
        t = time.time()

        # uses cv2.NORM_L1 for speed
        # matcher = cv2.DescriptorMatcher_create("BruteForce-L1")     # replaces "BruteForce" because L1 seems to give slightly better results. Necessary?
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

        # error messages for if features are missing
        if (featuresA is None) or (featuresB is None):
            print "Need to provide two sets of features"
            if (featuresA is None):
                print "FeaturesA missing"
            if (featuresB is None):
                print "featureB missing"
            # cv2.waitKey(0)
            return None
        if (featuresA is []) or (featuresB is []):
            print "Need to provide two sets of features"
            if (featuresA is None):
                print "FeaturesA missing"
            if (featuresB is None):
                print "featureB missing"
            # cv2.waitKey(0)
            return None

        # returns the two best matches found by the brute force matcher
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # rawMatches = matcher.match(featuresA,featuresB) Necessary?

        # prints time matches took to generate
        elapsed = time.time() - t
        print "Matching Feature Points: %f" % elapsed
        matches = []

        # loop over the raw matches
        t = time.time()
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # print len(featuresA)
        # print len(featuresB)

        # print len(matches), "Matches found"

        # computing a homography requires at least 4 matches, we use 10 to ensure a robust stitch.
        if len(matches) > 20:

            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
            if H is None:
                print "Homography failed: %d matches \n" % len(matches)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            elapsed = time.time() - t
            print "Computing Homography: %f Seconds" % elapsed
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        print "No valid Homography \n"
        return None

    # Description: draws matches on both images
    # Inputs: explicit self, first image, second image, keypoints of image1, keypoints of image2, status flag if previous function completed or not (unused)
    # Outputs: an image with the the the matches visualized
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

    # Description: serves to display the image as seen from a new virtual camera.
    # Inputs: (probably should be self, but works in current form), rotation matrix, translation matrix, image to process
    # Outputs: Processed image
    def changeView(unknown, R, t, in_image):
        ti = time.time()
        im_lim = in_image.shape
        # We assume that R is a rotation matrix with each row corresponding to the direction of that axis.
        H = np.append(R, t, 1)
        # find new image bounds
        corners = np.transpose(
            np.array([[0, 0, 1, 1], [im_lim[1], 0, 1, 1], [im_lim[1], im_lim[0], 1, 1], [0, im_lim[0], 1, 1]]))
        im_bounds = np.dot(H, corners)
        max_x = round(max(np.divide(im_bounds[0, :], im_bounds[2, :])))
        max_y = round(max(np.divide(im_bounds[1, :], im_bounds[2, :])))

        output = np.zeros([max_y + 1, max_x + 1, 3], dtype="uint8")
        # Apply transformation in 3D space
        for x in range(0, in_image.shape[1]):
            for y in range(0, in_image.shape[0]):
                in_pt = np.array([[x], [y], [1], [1]])
                out_pt = np.dot(H, in_pt)
                if out_pt[2] != 0:
                    out_x = int(round(out_pt[0] / out_pt[2]))
                    out_y = int(round(out_pt[1] / out_pt[2]))

                    if (out_x >= 0) & (out_y >= 0) & (out_x < max_x + 1) & (out_y < max_x + 1):
                        output[out_y, out_x, :] = in_image[y, x, :]

        elapsed = time.time() - ti
        print " \nWarping Viewpoint: %f Seconds \n " % elapsed
        return output

    # Description: Applies the projective transformation H to image A and then blends it with image B.
    # Inputs: The second image, the first image, the homography matrix
    # Outputs: result1, result2, mask1, mask2, coordinate shift as an array, the translation matrix
    def applyHomography(self, imageB, imageA, H): # should I swap these inputs? maybe
        # Detect The appropriate size for the resulting image.
        corners = np.array([[0, 0, 1], [0, imageA.shape[0], 1], [imageA.shape[1], 0, 1],
                            [imageA.shape[1], imageA.shape[0], 1]]).T

        # print H, corners
        img_bounds = np.dot(H, corners)

        # determines bounds and coordinate shift of resulting image
        x_bound = np.divide(img_bounds[0, :], img_bounds[2, :])
        y_bound = np.divide(img_bounds[1, :], img_bounds[2, :])
        x_shift = 0
        y_shift = 0
        if min(x_bound) < 0:
            x_shift = -int(min(x_bound))
        if min(y_bound) < 0:
            y_shift = -int(min(y_bound))

        shift_H = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]])
        x_bound = int(max(max(x_bound), imageB.shape[1]))
        y_bound = int(max(max(y_bound), imageB.shape[0]))

        # print error if image cannot be printed
        if (x_bound + x_shift > SIZE_BOUNDS[0]) or (y_bound + y_shift > SIZE_BOUNDS[1]):
            print x_bound, y_bound
            print "ERROR: Image Too Large"
            return 0, 0, 0, 0

        trans_matrix = np.dot(shift_H, H)

        # Warp Image A and place it in frame.
        imageB2 = np.pad(imageB, ((y_shift, 0), (x_shift, 0), (0, 0)), 'constant', constant_values=0)
        result2 = cv2.warpPerspective(imageA, trans_matrix,
                                      (x_bound + x_shift, y_bound + y_shift))
        # result2 = np.zeros([y_bound+y_shift,x_bound+x_shift,3]) Necessary?
        result1 = np.pad(imageB2,
                         ((0, y_bound + y_shift - imageB2.shape[0]), (0, x_bound + x_shift - imageB2.shape[1]), (0, 0)),
                         'constant', constant_values=0)

        mask1 = (result1 > 0).astype('int')
        mask2 = (result2 > 0).astype('int')

        return result1, result2, mask1, mask2, [x_shift, y_shift], trans_matrix

    # Description: locateSeam detects the location of the seam between maskA and maskB during blending.
    # Inputs: explicit self, Mask a is binary image if image A was greater than 0, image B is also a binary image, combined the two give x,y location for where
    # image A and image B were located (after transformation)
    # Outputs: Mask with contours drawn on
    def locateSeam(self, maskA, maskB):
        out_mask = np.zeros(maskA.shape).astype('uint8')
        contour_copy = np.zeros(maskA.shape).astype('uint8')
        contour_copy[:] = maskA[:].astype('uint8') * 255
        contour_copy = contour_copy.astype('uint8')
        im2, contours, hierarchy = cv2.findContours(contour_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_mask, contours, -1, (255, 255, 0), 1)
        out_mask = np.logical_and(out_mask, maskB).astype('float')
        return out_mask

    # Description: calculates the distance from point A to the closest nonzero point on maskB.
    # Inputs: explicit self, tuple of keypoints from image1, mask from image2 as array
    # Outputs: array of ints holding the appropriate distances
    def calcMinDistance(self, pointsA, maskB):
        output = np.zeros(len(pointsA))
        pointsB = np.nonzero(maskB)
        distance = np.zeros(len(pointsB[0]))

        for k in range(0, len(output)):
            pointA = pointsA[k]
            for i in range(0, len(distance)):
                distance[i] = ((pointA[0] - pointsB[0][i]) ** 2 + (pointA[1] - pointsB[1][i]) ** 2) ** (0.5)

            output[k] = np.min(distance)

        return output

    # Description: Shifts to images based on their homography and overlaps them
    # Inputs: Explicit self, image1, image2, the homography matrix
    # Outputs: results are both images (result 2) where A is transformed and shifted, and (result1) shifted B is simply shifted,
    # coordinate shift as a tuple of two ints (amount final image has to be shifted by so nothing is negative)
    def shiftImage(self, imageA, imageB, H):
        # Detect the appropriate size for the resulting image.
        corners = np.array([[0, 0, 1], [0, imageA.shape[0], 1], [imageA.shape[1], 0, 1],
                        [imageA.shape[1], imageA.shape[0], 1]]).T

        # print H, corners
        img_bounds = np.dot(H, corners)

        # determines bounds and coordinate shift of resulting image
        x_bound = np.divide(img_bounds[0, :], img_bounds[2, :])
        y_bound = np.divide(img_bounds[1, :], img_bounds[2, :])
        x_shift = 0
        y_shift = 0
        if min(x_bound) < 0:
            x_shift = -int(min(x_bound))
        if min(y_bound) < 0:
            y_shift = -int(min(y_bound))
        coord_shift = np.array([y_shift, x_shift])

        shift_H = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]])
        x_bound = int(max(max(x_bound), imageB.shape[1]))
        y_bound = int(max(max(y_bound), imageB.shape[0]))

        # prints error if resulting image cannot be displayed
        print "X Bound:", x_bound, "Y Bound:", y_bound, "x shift:", x_shift, "y shift:", y_shift
        if (x_bound + x_shift > SIZE_BOUNDS[0]) or (y_bound + y_shift > SIZE_BOUNDS[1]):
            print x_bound, y_bound
            print "ERROR: Image Too Large"
            return 0, 0, (0, 0)

        # Warp Image A and place it in frame.
        imageB2 = np.pad(imageB, ((y_shift, 0), (x_shift, 0), (0, 0)), 'constant', constant_values=0)
        result2 = cv2.warpPerspective(imageA, np.dot(shift_H, H),
                                  (x_bound + x_shift, y_bound + y_shift))

        result1 = np.pad(imageB2,
                     ((0, y_bound + y_shift - imageB2.shape[0]), (0, x_bound + x_shift - imageB2.shape[1]), (0, 0)),
                     'constant', constant_values=0)

        return result1, result2, coord_shift

#Description: Legacy code, unused, potentially remove
def reStitch(self, image1, image2, canvas, fgbg, seam, out_pos):
    # cv2.imshow("seam",np.seam)
    seam_points = np.nonzero(seam)  # Determine the point locations of the seam
    # print seam_points
    seam_bounds = [np.min(seam_points[0]), np.max(seam_points[0]), np.min(seam_points[1]), np.max(seam_points[1])]

    seam_bounds[0] = np.max([(seam_bounds[0] - SEAM_PAD), 0])
    seam_bounds[2] = np.max([seam_bounds[2] - SEAM_PAD, 0])
    seam_bounds[1] = np.min([seam_bounds[1] + SEAM_PAD, image1.shape[0]])
    seam_bounds[3] = np.min([seam_bounds[3] + SEAM_PAD, image1.shape[1]])

    # Foreground Re-Stitching
    fgmask = fgbg.apply(image2)  # Apply Background Subtractor
    out_frame = np.zeros(fgmask.shape)  # Generate correctly sized output_frame for foreground mask

    if (len(np.unique(fgmask)) <= 1):
        return canvas, fgbg

    # DEBUGGING: show fgmask before dilation
    # cv2.imshow('fgmask',fgmask)
    # Denoise by erosion, then use dilation to fill in holes
    # fgmask = cv2.erode(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS)
    # fgmask = cv2.dilate(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS+DILATION_LOOPS)

    # Find bounding rectangle of all moving contours.
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # DEBUGGING: Show drawn contours.
    # tmp_seam = 255*np.tile(seam[...,None],[1,1,3])
    # print tmp_seam.shape
    # tmp_show = cv2.drawContours(tmp_seam.astype('uint8'),contours,-1,(0,255,0),3)
    # cv2.imshow('contours',tmp_show)

    x = np.zeros(len(contours))
    y = np.zeros(len(contours))
    w = np.zeros(len(contours))
    h = np.zeros(len(contours))
    for i in range(0, len(contours)):
        if (i % 1 == 0):
            cnt = contours[i]

            x[i], y[i], w[i], h[i] = cv2.boundingRect(cnt)
            out_frame[y[i]:y[i] + h[i], x[i]:x[i] + w[i]] = (i + 1) * np.ones([h[i], w[i]])

        # cv2.imshow("Moving Objects",out_frame)
        # Create list of moving objects that cross the seam line.
    moving_objects = np.unique(out_frame * seam)
    print "Moving Objects: ", moving_objects

    # If there are objects that cross the seam line, we attempt to re-stitch those objects.
    if len(moving_objects) > 1:

        moving_objects = moving_objects[1:].astype('int')
        # For each object that crosses the seam:
        for i in moving_objects:
            # DEBUGGING: output to check which objects were detected
            print "object ", i, " detected"
            # cv2.imshow("Object Detected",(out_frame == i).astype('float'))

            # Print statements to help with testing.
            # print "object %d in seam" % moving_objects[i]
            # print x[i-1],y[i-1],w[i-1],h[i-1]
            # cv2.imshow('image1',image1[seam_bounds[0]:seam_bounds[1],seam_bounds[2]:seam_bounds[3]])
            # cv2.imshow('image2',image2[y[i-1]:y[i-1]+h[i-1],x[i-1]:x[i-1]+w[i-1]])

            # if the object is large enough for feature point detection to function appropriately.
            if (w[i - 1] > 25) & (h[i - 1] > 25):
                # print x[i-1],y[i-1],w[i-1],h[i-1]
                # image2 = cv2.rectangle(image2,(x[i-1].astype('int'),y[i-1].astype('int')),(x[i-1].astype('int')+w[i-1].astype('int'),y[i-1].astype('int')+h[i-1].astype('int')),(0,255,0),2)

                mask = image1 > 0
                mask = mask[y[i - 1]:y[i - 1] + h[i - 1], x[i - 1]:x[i - 1] + w[i - 1]]

                # Stitch together seam area with object bounding box.
                r1, vis_tmp, Htemp, m1, m2, coord_shift = self.stitch(
                    [image1[seam_bounds[0]:seam_bounds[1], seam_bounds[2]:seam_bounds[3]],
                     image2[y[i - 1]:y[i - 1] + h[i - 1], x[i - 1]:x[i - 1] + w[i - 1]]], showMatches=True,
                    reStitching=True, seam=mask)
                y_shift = coord_shift[0]
                x_shift = coord_shift[1]
                # If the stitch was successful.
                if r1 is not 0:
                    print "Re-stitched"
                    # Show the stitched section.
                    # cv2.imshow('Stitched small', r1)
                    # cv2.waitKey(0)
                    out_pos_item = [out_pos[0] + seam_bounds[0] - y_shift, out_pos[1] + seam_bounds[2] - x_shift]

                    if (out_pos_item[0] < 0):
                        print "Object located out of image"
                        r1 = r1[-out_pos_item[0]:, :, :]
                        m1 = m1[-out_pos_item[0]:, :, :]
                        m2 = m2[-out_pos_item[0]:, :, :]
                        out_pos_item[0] = 0

                    if (out_pos_item[1] < 0):
                        print "Object located out of image"
                        r1 = r1[:, -out_pos_item[1], :]
                        m1 = m1[:, -out_pos_item[1], :]
                        m2 = m2[:, -out_pos_item[1], :]
                        out_pos[1] = 0

                    tmp_result = canvas[out_pos_item[0]:out_pos_item[0] + r1.shape[0],
                                 out_pos_item[1]:out_pos_item[1] + r1.shape[1]]

                    print coord_shift
                    r1 = r1[0:tmp_result.shape[0], 0:tmp_result.shape[1], :]
                    m1 = m1[0:tmp_result.shape[0], 0:tmp_result.shape[1], :]
                    m2 = m2[0:tmp_result.shape[0], 0:tmp_result.shape[1], :]

                    print "Out Pos:", out_pos_item
                    print "tmp_result:", tmp_result.shape, "canvas slice:", canvas[
                                                                            out_pos_item[0]:out_pos_item[0] + r1.shape[
                                                                                0],
                                                                            out_pos_item[1]:out_pos_item[1] + r1.shape[
                                                                                1]].shape, "r1:", r1.shape, "m1:", m1.shape, "m2:", m2.shape
                    canvas[out_pos_item[0]:out_pos_item[0] + r1.shape[0],
                    out_pos_item[1]:out_pos_item[1] + r1.shape[1]] = m2 * np.logical_not(m1) * r1 + np.logical_not(
                        m2) * tmp_result
                    canvas[out_pos_item[0], out_pos_item[1], :] = [255, 0, 0]
                    print coord_shift, out_pos_item

                    # small_result = canvas[out_pos_item[0]:out_pos_item[0]+r1.shape[0],out_pos_item[1]:out_pos_item[1]+r1.shape[1]]
                    # small_result = r1
                    # cv2.imshow('image1',image1[seam_bounds[0]:seam_bounds[1],seam_bounds[2]:seam_bounds[3]])
                    # cv2.imshow('m1',m1.astype('float'))
                    # cv2.imshow('m2',m2.astype('float'))
                    # cv2.imshow('re-stitched', small_result.astype('uint8'))
                    # cv2.waitKey(0)

                    cv2.imshow('features', vis_tmp)
                    cv2.imshow('result', r1)
                # cv2.waitKey(0)
                # if cv2.waitKey(0) & 0xFF == ord('s'):
                #    print "Saving Feature Matches"
                #    cv2.imwrite('Vis.jpg',vis_tmp)

        # cv2.imshow('bounding boxes',image2)
    return canvas, fgbg
