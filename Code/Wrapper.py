#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano:

"""

# Code starts here:

import numpy as np
import cv2
import argparse
import os
import copy
from skimage.feature import peak_local_max 
import math
from PIL import Image 

# Add any python libraries here


class AutoPano():
    def __init__(self,NumFeatures,ImagePath) -> None:
        print("initialized",NumFeatures,ImagePath)
        self.images = []
        self.gray_images = []
        self.corner_strn = []
        self.ext_harris_img = []
        self.feature_images = []
        self.show_images = False


    def read_images(self,path):
        images = []
        # gray_images = []
        if not os.path.exists(path):
            return []

        files = sorted(os.listdir(path=path))
        for file in files:
            p = path+'/'+file
            try :
                img = cv2.imread(p)
                gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                images.append(img)

                
            except:
                print("Kuch to Gadbad hai Daya!!")
        return images
            
    def extract_Harriscorner(self,imgs,gimgs):
        images = copy.deepcopy(imgs)
        gray_images = copy.deepcopy(gimgs)
        corner_strn = []
        ext_harris_img = []
        for i,img in enumerate(gray_images):
            corner_strn_loc = cv2.cornerHarris(img,blockSize=2,ksize=3,k=0.04)
            print("Corner strn shape : ",corner_strn_loc.shape)
            # corner_strn[corner_strn<0.01*corner_strn.max()] = 0
            corner_strn.append(corner_strn_loc)
            # cv2.imshow("dst",corner_strn_loc)
            # if cv2.waitKey(0) & 0xff == 27:
                # cv2.destroyAllWindows()
            dst = cv2.dilate(corner_strn_loc,None)
            copyimg = images[i].copy()
            # copyimg = images[i]
            copyimg[dst>0.001*dst.max()] = [0,0,255]
            ext_harris_img.append(copyimg)
            disp = np.concatenate((images[i],copyimg),axis=1)
            if self.show_images:
                cv2.imshow("dst",disp)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

        return corner_strn

    def extract_shi(self,imgs):
        images = copy.deepcopy(imgs)
        corner_images = []
        for i,img in enumerate(images):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            corners_strn = cv2.goodFeaturesToTrack(img,200,0.01,10)
            corners_strn = np.int64(corners_strn)
            corner_images.append(corners_strn)
            for c in corners_strn:
                x,y = c.ravel()
                cv2.circle(img,(x,y),3,(0,0,255),-1)
            
        return corner_images

    def anms(self,num_features,imgs,gimgs,crn_stn):

        images = copy.deepcopy(imgs)
        gray_images = copy.deepcopy(gimgs)
        corner_strn = copy.deepcopy(crn_stn)
        corner_images = []
        for pos,(img,corner) in enumerate(zip(gray_images,corner_strn)):

            local_max = peak_local_max(image=corner,num_peaks=1000,min_distance=10)
            r = [np.Infinity for i in range(local_max.shape[0])]
            ed = 0
            peak_loc = np.zeros((len(r),2))
            for i in range(len(r)):
                # ed = np.Infinity
                for j in range(len(r)):
                    xi,yi = local_max[i]
                    xj,yj = local_max[j]
                    if corner[xj,yj] > corner[xi,yi]:
                        ed = np.square(xj-xi) + np.square(yj-yi)
                    if r[i] > ed:
                        r[i] = ed
                        peak_loc[i] = [yi,xi]
                    
            sorted_index = np.argsort(r)
            sorted_index = np.flip(sorted_index)
            num_features = len(r) if num_features>len(r) else num_features
            sorted_peak_loc = [peak_loc[i] for i in sorted_index[:num_features]]

            copyimg = copy.deepcopy(images[pos])
            copyimg = np.array(copyimg)
            corner_images.append(sorted_peak_loc)
            if self.show_images:
                for i in sorted_peak_loc:
                    cv2.circle(copyimg,(int(i[0]),int(i[1])),3,(255,0,0),-1)
                cv2.imshow("ANMS",copyimg)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()
        return corner_images

    def feature_descp(self,corners,patch_size,imgs):
        feature_images = []
        images = copy.deepcopy(imgs)
        selected_corners = []
        for image,corner in zip(images,corners):
            img = copy.deepcopy(image)
            gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray_image = np.pad(gray_image,[(20,20),(20,20)])
            vectors = []
            loc = []
            w,h,ch = img.shape
            for c in corner:
                # x,y = c.ravel()
                y,x = np.int32(c)

                if (x-(patch_size//2)>0) and (x+(patch_size//2) < w) and (y-(patch_size//2) > 0) and (y+(patch_size//2) < h):
                # patch = gray_image[(x-(patch_size //2)):(x+1+(patch_size//2)),(y-(patch_size//2)):(y+1+(patch_size//2))]
                    patch = gray_image[x:(x+41),y:(y+41)]
                    filter = cv2.GaussianBlur(patch,(5,5),0)
                    sub = cv2.resize(filter,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_CUBIC)
                    vector = sub.reshape(-1)
                    # if len(vector) != 64:
                        # print("Problem")
                    vector = (vector-np.mean(vector))/np.std(vector)
                    vectors.append(vector)
                    loc.append([y,x])
            selected_corners.append(loc)
            feature_images.append(vectors)
        return selected_corners,feature_images

    def feature_match(self,corners,fimgs):

        feature_images = copy.deepcopy(fimgs)
        corner_0 = copy.deepcopy(corners[0])
        image_feat_match = []
        feature_pairs = []
        for i,corner_1 in enumerate(corners[1:]):
            image_feat_match = []
            for n,feat_1 in enumerate(feature_images[0]):
                dis = []
                for j,feat_2 in enumerate(feature_images[i+1]):
                    # if len(feat_2) != 64 or len(feat_1) != 64:
                        # print("gadbad")
                    dis.append(np.sum((feat_1-feat_2)**2))
                
                sort_dis = np.argsort(dis)
                # print("Min Value: ",dis[sort_dis[0]],dis[sort_dis[1]])
                if dis[sort_dis[0]]/dis[sort_dis[1]]  < 0.7 :
                    image_feat_match.append([corner_0[n],corner_1[sort_dis[0]]])
                # image_feat_match.append([corner_0[n],corner_1[sort_dis[0]]])
                




            feature_pairs.append(np.array(image_feat_match))
            
        return feature_pairs
    
    def makeImageSizeSame(self,imgs):
        images = imgs.copy()
        sizes = []
        for image in images:
            x, y, ch = image.shape
            sizes.append([x, y, ch])

        sizes = np.array(sizes)
        x_target, y_target, _ = np.max(sizes, axis = 0)

        images_resized = []

        for i, image in enumerate(images):
            image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
            image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
            images_resized.append(image_resized)

        return images_resized
    
    def displaymatches(self,matched_pairs,images,msg):
        image_0 = copy.deepcopy(images[0])
        image_1 = copy.deepcopy(images[1])

        corner_0 = copy.deepcopy(matched_pairs[:,0])
        corner_1 = copy.deepcopy(matched_pairs[:,1])
        corner_1[:,0] += image_0.shape[1]
        h0, w0, _ = image_0.shape
        h1, w1, _ = image_1.shape

        image_0,image_1 = self.makeImageSizeSame(imgs=[image_0,image_1])
        
        comb_image = np.concatenate((image_0,image_1),axis=1)

        for (x0,y0),(x1,y1) in zip(corner_0,corner_1):

            cv2.line(comb_image,(x0,y0),(x1,y1),(0,255,0),1)
        cv2.imshow(msg,comb_image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def homography(self,matched_pairs):
        set_of_pairs = 4
        corner_0 = matched_pairs[:,0]
        corner_1 = matched_pairs[:,1]
        n = corner_0.shape[0]
        num_inliers = 0
        final_indices = None
        H_best = np.zeros([3,3])
        H = np.identity(3)
        for i in range(2000):
            n_rndm = np.random.choice(n,size=set_of_pairs)

            sel_corner_0 = copy.deepcopy(corner_0[n_rndm])
            sel_corner_1 = copy.deepcopy(corner_1[n_rndm])

            # H = cv2.getPerspectiveTransform(sel_corner_0,sel_corner_1)
            H,_ = cv2.findHomography(np.float32(sel_corner_0),np.float32(sel_corner_1))

            corner_0_z = np.vstack((corner_0[:,0],corner_0[:,1],np.ones([n])))
            if H is None:
                H = np.zeros([3,3])
            Hp0 = np.dot(H,corner_0_z)

            norm_Hp0 = Hp0/np.expand_dims(Hp0[2,:],axis=0)

            # ssd = 0
            sel_ind = []
            for i,(x,y) in enumerate(zip(norm_Hp0[0,:],norm_Hp0[1,:])):
                ssd = ( (corner_1[i][0] - x)**2 + (corner_1[i][1]-y)**2   )
                # print("ssd-ransac",ssd)
                if ssd < 20:
                    sel_ind.append(i)
            
            if len(sel_ind) > num_inliers:
                num_inliers = len(sel_ind)
                final_indices = sel_ind
                H_best = H
        filt_corner_0 = corner_0[final_indices]
        filt_corner_1 = corner_1[final_indices]

        filtered_pairs = np.stack((filt_corner_0,filt_corner_1),axis=1)

        return filtered_pairs,H_best
    
    
    def homogeneous_coordinate(self,coordinate): # Not Used
        x = coordinate[0]/coordinate[2]
        y = coordinate[1]/coordinate[2]
        return x, y
    
    
    def warp(self,image, homography): # Not Used
        print("Warping is started.")

        image_array = np.array(image)
        row_number, column_number = int(image_array.shape[0]), int(image_array.shape[1])

        up_left_cor_x, up_left_cor_y = self.homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
        up_right_cor_x, up_right_cor_y = self.homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
        low_left_cor_x, low_left_cor_y = self.homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
        low_right_cor_x, low_right_cor_y = self.homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))

        x_values = [up_left_cor_x, up_right_cor_x, low_right_cor_x, low_left_cor_x]
        y_values = [up_left_cor_y, up_right_cor_y, low_left_cor_y,  low_right_cor_y]
        print("x_values: ", x_values, "\n y_values: ", y_values)

        offset_x = math.floor(min(x_values))
        offset_y = math.floor(min(y_values))
        print("offset_x: ", offset_x, "\t size_y: ", offset_x)

        max_x = math.ceil(max(x_values))
        max_y = math.ceil(max(y_values))

        size_x = max_x - offset_x
        size_y = max_y - offset_y
        print("size_x: ", size_x, "\t size_y: ", size_y)

        homography_inverse = np.linalg.inv(homography)
        print("Homography inverse: ", "\n", homography_inverse)

        result = np.zeros((size_y, size_x, 3))

        for x in range(size_x):
            for y in range(size_y):
                point_xy = self.homogeneous_coordinate(np.dot(homography_inverse, [[x+offset_x], [y+offset_y], [1]]))
                point_x = int(point_xy[0])
                point_y = int(point_xy[1])

                if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number):
                    result[y, x, :] = image_array[point_y, point_x, :]

        print("Warping is completed.")
        return result, offset_x, offset_y
    

    def blending2images(self,base_array, image_array, offset_x, offset_y): # Not Used
        print("Blending two images is started.")

        image_array = np.array(image_array)
        base_array = np.array(base_array)

        rows_base, columns_base = int(base_array.shape[0]), int(base_array.shape[1])
        rows_image, columns_image = int(image_array.shape[0]), int(image_array.shape[1])

        print("Column number of base: ", columns_base, "\t Row number of base: ", rows_base)
        print("Column number of image: ", columns_image, "\t Row number of image: ", rows_image)

        x_min = 0
        if offset_x>0:
            x_max = max([offset_x+columns_image, columns_base])
        else:
            x_max = max([-offset_x + columns_base, columns_image])

        y_min = 0
        # note that offset_y was always negative in this assignment.
        if offset_y>0:
            y_max = max([rows_image+offset_y, rows_base])
        else:
            y_max = max([rows_base-offset_y, rows_image])

        size_x = x_max - x_min
        size_y = y_max - y_min

        print("size_x: ", size_x, "\t size_y: ", size_y)
        blending = np.zeros((size_y, size_x, 3))

        # right to left image stitching
        # offset_x -=
        print("offset_x :",offset_x)
        if offset_x > 0:
            if offset_y >0 :
                blending[offset_y:rows_image+offset_y, offset_x:columns_image+offset_x, :] = image_array[:, :, :]
            # blending[:rows_image, :columns_image, :] = image_array[:, :, :]

                blending[:rows_base, :columns_base, :] = base_array[:, :, :]
            else:
                blending[:rows_image, offset_x:columns_image+offset_x, :] = image_array[:, :, :]
                # blending[:rows_image, :columns_image, :] = image_array[:, :, :]

                blending[-offset_y:rows_base-offset_y, :columns_base, :] = base_array[:, :, :]
        # left to right image stitching
        else:
            if offset_y>0:
                blending[offset_y:rows_image+offset_y, :columns_image, :] = image_array[:, :, :]
                blending[:rows_base, -offset_x:columns_base-offset_x, :] = base_array[:, :, :]
            else:
                blending[:rows_image, :columns_image, :] = image_array[:, :, :]
                blending[-offset_y:rows_base-offset_y, -offset_x:columns_base-offset_x, :] = base_array[:, :, :]

        print("Blending is completed.")
        return blending
    
    def cropImageRect(self,image):
    
        img = image.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        _,thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(contours[len(contours)-1])
        crop = img[y:y+h,x:x+w]

        return crop
    
    def blending(self,images,H):
        image_0 = copy.deepcopy(images[0])
        image_1 = copy.deepcopy(images[1])
        h0,w0,_ = image_0.shape
        h1,w1,_ = image_1.shape

        pnts_0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1,1,2)
        pnts_0_trans = cv2.perspectiveTransform(pnts_0,H)
        pnts_1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)

        pnts_merged = np.concatenate((pnts_0_trans,pnts_1),axis=0)
        points_on_merged_images = []
        for p in range(len(pnts_merged)):
            points_on_merged_images.append(pnts_merged[p].ravel())

        points_on_merged_images = np.array(points_on_merged_images)

        x_min, y_min = np.int0(np.min(points_on_merged_images, axis = 0))
        x_max, y_max = np.int0(np.max(points_on_merged_images, axis = 0))

        print("min, max")
        print(x_min, y_min)
        print(x_max, y_max)
        
        H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

        image0_transformed_and_stitched = cv2.warpPerspective(image_0, np.dot(H_translate, H), (x_max-x_min, y_max-y_min))

        images_stitched = image0_transformed_and_stitched.copy()
        images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = image_1

        # mask = np.ones(image_1.shape).astype('uint8') * 255


        # cv2.circle(image0_transformed_and_stitched,(int(-x_min),int(-y_min)),3,(255,0,0),2)
        # cv2.circle(image0_transformed_and_stitched,(int(x_max),int(y_max)),3,(0,255,0),2)

        # cv2.imshow("mask",mask)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()
        # center = (-x_min+(w1//2),-y_min+(h1//2))

        # blended = cv2.seamlessClone(src=image_1,dst=image0_transformed_and_stitched,mask=mask,p=center,flags=cv2.NORMAL_CLONE)
        
        # cv2.imshow("blended",blended)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()

        indices = np.where(image_1 == [0,0,0])
        y = indices[0] + -y_min 
        x = indices[1] + -x_min 

        images_stitched[y,x] = image0_transformed_and_stitched[y,x]

        return images_stitched




def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=1000, help='Number of best features to extract from each image, Default:1000')
    Parser.add_argument('--ImagePath',default='/home/smit/computer_vision/smshah1_p1/Phase1/Data/Train/Set1',help="Path of the Image(Data) folder")
    # Parser.add_argument('--SavePath',default='/home/smit/computer_vision/smshah1_p1/Phase1/Data/Train/Set2/result',help="Path of the Image(Data) folder")
    Parser.add_argument('--ShowImages',default=False,help="Will Show intermediate images if True, Default: False")


    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    ImagePath = Args.ImagePath
    ShowImages = Args.ShowImages
    SavePath = ImagePath + "/result"
    obj = AutoPano(NumFeatures,ImagePath)
    lst_imgs = obj.read_images(ImagePath)
    # lst_imgs = copy.deepcopy(lst_lst_imgs)
    itr=0
    stitched_images = []
    # iterations = math.ceil(len(lst_imgs)/2)
    iterations = len(lst_imgs)-1
    obj.show_images = ShowImages
    while itr < iterations:
        lst_imgs = obj.read_images(ImagePath)
        i=0
        if iterations >2:
            lst_imgs.reverse()
        while i < len(lst_imgs):
            if i+2 > len(lst_imgs):
                break
            imgs = lst_imgs[i:i+2]
            imgs.reverse()
            gimgs = []
            gimgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in imgs]
            corn_strn = obj.extract_Harriscorner(imgs=imgs,gimgs=gimgs)
            corner_images = obj.anms(num_features=NumFeatures,imgs=imgs,gimgs=gimgs,crn_stn=corn_strn)
            selected_corners,f_imgs = obj.feature_descp(corners=corner_images,patch_size=40,imgs=imgs)
            matched_pairs = obj.feature_match(corners=selected_corners,fimgs=f_imgs)
            if ShowImages:

                obj.displaymatches(matched_pairs=matched_pairs[0],images=imgs,msg="Matching Pairs")

            filtered_match,H_best = obj.homography(matched_pairs=matched_pairs[0])
            if ShowImages:
                obj.displaymatches(matched_pairs=filtered_match,images=imgs,msg="Matching Pairs after RANSAC")
            
            a = len(np.unique(filtered_match[:,0]))
            b = len(np.unique(filtered_match[:,1]))

            if b/a < 0.6 or b/a > 1.6:
                i+=1
                print("Cannot Not Stitch because not enough common features available")
                continue

            stitched_image = obj.blending(images=imgs,H=H_best)
            stitched_image = obj.cropImageRect(image=stitched_image)
            if ShowImages:
                cv2.imshow("Stitched Image",stitched_image)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()
            new_dir_path = ImagePath + "/temp"+str(itr)
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            save_path = new_dir_path+"/stitched_image_"+str(i)+".png"
            cv2.imwrite(save_path,stitched_image)
            stitched_images.append(stitched_image)
            if iterations>2 and itr==4:
                if len(lst_imgs)<=3:
                    i+=1
                else:
                    i = i+2
            else:
                i = i+1
        
        ImagePath = ImagePath+"/temp"+str(itr)
        itr+=1

    if not os.path.exists(SavePath):
                os.makedirs(SavePath)
    SavePath = SavePath + "/Pano.png"
    cv2.imwrite(SavePath,stitched_images[-1])


    print("Blended image is generated.")

if __name__ == "__main__":
    main()
