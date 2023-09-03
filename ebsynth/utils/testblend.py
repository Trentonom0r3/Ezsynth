import cv2
import numpy as np

class StylizedImageBlender:
    def __init__(self, stylized_image1, stylized_image2, mask):
        self.image1 = stylized_image1
        self.image2 = stylized_image2
        self.gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        self.mask = mask

    def _extract_keypoints_descriptors(self, gray_image):
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
        return keypoints, descriptors

    def _match_keypoints(self, descriptors1, descriptors2):
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        
        # Check if there are enough keypoints to match
        if descriptors1.shape[0] < 2 or descriptors2.shape[0] < 2:
            return np.array([]), np.array([])

        k_value = min(descriptors1.shape[0], descriptors2.shape[0], 2)
        
        matches = flann.knnMatch(descriptors1, descriptors2, k=k_value)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        src_pts = np.float32([self.keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return src_pts, dst_pts
    
    def _calculate_local_homography(self, src_pts, dst_pts):
        if len(src_pts) < 4 or len(dst_pts) < 4:  # Need at least 4 points for homography
            return None

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return homography

    def _warp_local_image(self, homography, region, source_image):
        if homography is None:  # Check for a valid homography matrix
            return None

        (x, y, w, h) = region
        local_image = source_image[y:y+h, x:x+w]
        
        try:
            aligned_image = cv2.warpPerspective(local_image, homography, (w, h))
        except cv2.error as e:
            print(f"Error in warpPerspective: {e}")
            return None
        
        return aligned_image


    def blend_images(self):
        grid_size = 50
        rows, cols, _ = self.image1.shape
        # Initialize aligned_image as a copy of image2
        aligned_image = np.copy(self.image2)

        for y in range(0, rows, grid_size):
            for x in range(0, cols, grid_size):
                region = (x, y, grid_size, grid_size)
                local_mask = self.mask[y:y+grid_size, x:x+grid_size]

                # Decide which image to warp based on the mask
                white_pixels = np.sum(local_mask == 255)
                black_pixels = np.sum(local_mask == 0)
                
                if white_pixels > black_pixels:
                    # Warp image2 to match image1 in this region
                    source_image = self.image2
                    local_gray1 = self.gray1[y:y+grid_size, x:x+grid_size]
                    local_gray2 = self.gray2[y:y+grid_size, x:x+grid_size]
                else:
                    # Warp image1 to match image2 in this region
                    source_image = self.image1
                    local_gray1 = self.gray2[y:y+grid_size, x:x+grid_size]
                    local_gray2 = self.gray1[y:y+grid_size, x:x+grid_size]
                
                self.keypoints1, descriptors1 = self._extract_keypoints_descriptors(local_gray1)
                self.keypoints2, descriptors2 = self._extract_keypoints_descriptors(local_gray2)

                if descriptors1 is not None and descriptors2 is not None:
                    src_pts, dst_pts = self._match_keypoints(descriptors1, descriptors2)
                    if len(src_pts) >= 4 and len(dst_pts) >= 4:
                        homography = self._calculate_local_homography(src_pts, dst_pts)
                        aligned_local_image = self._warp_local_image(homography, region, source_image)
                        if aligned_local_image is not None:
                            aligned_image[y:y+grid_size, x:x+grid_size] = aligned_local_image

        mask = self.mask
        blended_result = cv2.seamlessClone(self.image1, aligned_image, mask, (cols // 2, rows // 2), cv2.NORMAL_CLONE)
        
        return blended_result
    
    def naive_alpha_blend(self):
        alpha = 0.5  # You can adjust this value for different blending strengths
        
        #expand self mask into 3 channel for np.where
        mask = np.expand_dims(self.mask, axis=2)
        blended_naive = cv2.seamlessClone(self.image1, self.image2, mask, (self.image1.shape[1] // 2, self.image1.shape[0] // 2), cv2.NORMAL_CLONE)
        return blended_naive
    
    def compare_blend_methods(self):
        blended_feature_based = self.blend_images()
        blended_naive = self.naive_alpha_blend()
        
        comparison = np.concatenate((blended_feature_based, blended_naive), axis=1)
        
        return comparison
    

