import cv2
import os
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if im.max()>10:

        im = np.float32(im)/255

    im_pyramid = []

    for i in levels:

        sigma_ = sigma0*k**i 

        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))

    im_pyramid = np.stack(im_pyramid, axis=-1)

    return im_pyramid







def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):

    '''

    Produces DoG Pyramid

    Inputs

    Gaussian Pyramid - A matrix of grayscale images of size

                        [imH, imW, len(levels)]

    levels      - the levels of the pyramid where the blur at each level is

                   outputs

    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

                   created by differencing the Gaussian Pyramid input

    '''

    ################

    # TO DO ...

    # compute DoG_pyramid here



    DoG_pyramid = []

    DoG_levels = levels[1:]

    for i in range(1, len(DoG_levels)+1):

        DoG_pyramid.append(gaussian_pyramid[:,:,i] - gaussian_pyramid[:,:,i-1])



    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)

    return DoG_pyramid, DoG_levels



def computePrincipalCurvature(DoG_pyramid):

    '''

    Takes in DoGPyramid generated in createDoGPyramid and returns

    PrincipalCurvature,a matrix of the same size where each point contains the

    curvature ratio R for the corre-sponding point in the DoG pyramid

    

    INPUTS

        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    

    OUTPUTS

        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 

                          point contains the curvature ratio R for the 

                          corresponding point in the DoG pyramid

    '''

    principal_curvature = np.zeros((DoG_pyramid.shape[0], DoG_pyramid.shape[1], DoG_pyramid.shape[2]))

    ##################

    # TO DO ...

    # Compute principal curvature here

    for i in range(0, DoG_pyramid.shape[2]):

        sobelx = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,1,0,ksize=3)

        sobely = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,0,1,ksize=3)



        sobelxx = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=3)

        sobelyy = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=3)

        sobelxy = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=3)

        sobelyx = cv2.Sobel(sobely,cv2.CV_64F,1,0,ksize=3)

        traceH = np.square(np.add(sobelxx,sobelyy))

        detH = np.subtract(np.multiply(sobelxx, sobelyy), np.multiply(sobelxy,sobelyx))

        principal_curvature[:,:,i] = np.divide(traceH, detH)

        principal_curvature[:,:,i] = np.nan_to_num(principal_curvature[:,:,i])

    return principal_curvature



def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,

        th_contrast=0.03, th_r=12):

    '''

    Returns local extrema points in both scale and space using the DoGPyramid



    INPUTS

        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

        DoG_levels  - The levels of the pyramid where the blur at each level is

                      outputs

        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the

                      curvature ratio R

        th_contrast - remove any point that is a local extremum but does not have a

                      DoG response magnitude above this threshold

        th_r        - remove any edge-like points that have too large a principal

                      curvature ratio

     OUTPUTS

        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both

               scale and space, and also satisfies the two thresholds.

    '''

    imh, imw, iml = DoG_pyramid.shape

    extremaTensor = np.zeros((11,imh,imw,iml))

    for layer in range(0, iml):

        temp_pyramid = np.pad(DoG_pyramid[:,:,layer],(1,1),mode='constant',constant_values=0)

        extremaTensor[0,:,:,layer] = np.roll(temp_pyramid,1,axis=1)[1:-1,1:-1] #right

        extremaTensor[1,:,:,layer] = np.roll(temp_pyramid,-1,axis=1)[1:-1,1:-1] #left

        extremaTensor[2,:,:,layer] = np.roll(temp_pyramid,1,axis=0)[1:-1,1:-1] #down

        extremaTensor[3,:,:,layer] = np.roll(temp_pyramid,-1,axis=0)[1:-1,1:-1] #up

        extremaTensor[4,:,:,layer] = np.roll(np.roll(temp_pyramid, 1, axis=1),1,axis=0)[1:-1,1:-1] #right,down

        extremaTensor[5,:,:,layer] = np.roll(np.roll(temp_pyramid, -1, axis=1),1,axis=0)[1:-1,1:-1] #left,down

        extremaTensor[6,:,:,layer] = np.roll(np.roll(temp_pyramid, -1, axis=1),-1,axis=0)[1:-1,1:-1] #left,up

        extremaTensor[7,:,:,layer] = np.roll(np.roll(temp_pyramid, 1, axis=1),-1,axis=0)[1:-1,1:-1] #right,up

        if layer == 0:

            extremaTensor[9,:,:,layer] = DoG_pyramid[:,:,layer+1] #layer above

        elif layer == iml-1:

            extremaTensor[8,:,:,layer] = DoG_pyramid[:,:,layer-1] #layer below

        else:

            extremaTensor[8,:,:,layer] = DoG_pyramid[:,:,layer-1] #layer below

            extremaTensor[9,:,:,layer] = DoG_pyramid[:,:,layer+1] #layer above

        extremaTensor[10,:,:,layer] = DoG_pyramid[:,:,layer]



    extremas = np.argmax(extremaTensor, axis=0)

    extremaPoints = np.argwhere(extremas==10)

    locsDoG = []



    for point in extremaPoints:

        if np.absolute(DoG_pyramid[point[0],point[1],point[2]]) > th_contrast and principal_curvature[point[0],point[1],point[2]] < th_r:

            point = [point[1], point[0], point[2]]

            locsDoG.append(point)



    locsDoG = np.stack(locsDoG, axis=-1)

    locsDoG = locsDoG.T

    return locsDoG



def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 

                th_contrast=0.03, th_r=12):

    '''

    Putting it all together



    Inputs          Description

    --------------------------------------------------------------------------

    im              Grayscale image with range [0,1].



    sigma0          Scale of the 0th image pyramid.



    k               Pyramid Factor.  Suggest sqrt(2).



    levels          Levels of pyramid to construct. Suggest -1:4.



    th_contrast     DoG contrast threshold.  Suggest 0.03.



    th_r            Principal Ratio threshold.  Suggest 12.



    Outputs         Description

    --------------------------------------------------------------------------



    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema

                    in both scale and space, and satisfies the two thresholds.



    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))

    '''

    ##########################

    # TO DO ....

    # compupte gauss_pyramid, gauss_pyramid here

    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)

    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)

    principal_curvature = computePrincipalCurvature(DoG_pyramid)

    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,th_contrast,th_r)

    return locsDoG, gauss_pyramid




def makeTestPattern(patch_width=9, nbits=256):

    '''

    Creates Test Pattern for BRIEF



    Run this routine for the given parameters patch_width = 9 and n = 256



    INPUTS

    patch_width - the width of the image patch (usually 9)

    nbits      - the number of tests n in the BRIEF descriptor



    OUTPUTS

    compareX and compareY - LINEAR indices into the patch_width x patch_width image 

                            patch and are each (nbits,) vectors. 

    '''

    #############################

    # TO DO ...

    # Generate testpattern here

    lin_combinations = patch_width**2

    # random choice

    compareX = np.random.choice(lin_combinations, nbits).reshape((nbits,1))

    compareY = np.random.choice(lin_combinations, nbits).reshape((nbits,1))

    test_pattern_file = '../results/testPattern.npy'

    if not os.path.isdir('../results'):

        os.mkdir('../results')

    np.save(test_pattern_file, [compareX, compareY])

    return  compareX, compareY



# load test pattern for Brief

test_pattern_file = '../results/testPattern.npy'

if os.path.isfile(test_pattern_file):

    # load from file if exists

    compareX, compareY = np.load(test_pattern_file)

else:

    # produce and save patterns if not exist

    compareX, compareY = makeTestPattern(9,256)

    if not os.path.isdir('../results'):

        os.mkdir('../results')

    np.save(test_pattern_file, [compareX, compareY])



def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,

    compareX, compareY):

    '''

    Compute Brief feature

     INPUT

     locsDoG - locsDoG are the keypoint locations returned by the DoG

               detector.

     levels  - Gaussian scale levels that were given in Section1.

     compareX and compareY - linear indices into the 

                             (patch_width x patch_width) image patch and are

                             each (nbits,) vectors.

    

    

     OUTPUT

     locs - an m x 3 vector, where the first two columns are the image

    		 coordinates of keypoints and the third column is the pyramid

            level of the keypoints.

     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number

            of valid descriptors in the image and will vary.

    '''

    ##############################

    # TO DO ...

    # compute locs, desc here

    desc = []

    locs = []

    for point in locsDoG:

        layer = point[2]

        im = gaussian_pyramid[:,:,layer]

        x = point[1]

        y = point[0]

        impatch = im[x-4:x+5,y-4:y+5]

        P = impatch.transpose().reshape(-1)

        if P.shape[0] < 81:

            continue

        else:

            im_desc = []

            for (x,y) in zip(compareX, compareY):

                if P[x] < P[y]:

                    im_desc.append(1)

                else:

                    im_desc.append(0)

            if len(im_desc) > 0:

                desc.append(im_desc)

                locs.append(point)



    locs = np.stack(locs, axis=-1)

    desc = np.stack(desc, axis=-1)

    locs = locs.T

    desc = desc.T

    return locs, desc



def briefLite(im):

    '''

    INPUTS

    im - gray image with values between 0 and 1



    OUTPUTS

    locs - an m x 3 vector, where the first two columns are the image coordinates 

            of keypoints and the third column is the pyramid level of the keypoints

    desc - an m x n bits matrix of stacked BRIEF descriptors. 

            m is the number of valid descriptors in the image and will vary

            n is the number of bits for the BRIEF descriptor

    '''

    ###################

    # TO DO ...

    locsDoG, gauss_pyramid = DoGdetector(im)

    DoG_pyramid, levels = createDoGPyramid(gauss_pyramid)

    test_pattern_file = '../results/testPattern.npy'

    if os.path.isfile(test_pattern_file):

        compareX, compareY = np.load(test_pattern_file)

    locs, desc = computeBrief(im, DoG_pyramid, locsDoG, np.sqrt(2), levels, compareX, compareY)

    return locs, desc



def briefMatch(desc1, desc2, ratio=0.8):

    '''

    performs the descriptor matching

    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.

                                n is the number of bits in the brief

    outputs : matches - p x 2 matrix. where the first column are indices

                                        into desc1 and the second column are indices into desc2

    '''

    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')

    # find smallest distance

    ix2 = np.argmin(D, axis=1)

    d1 = D.min(1)

    # find second smallest distance

    d12 = np.partition(D, 2, axis=1)[:,0:2]

    d2 = d12.max(1)

    r = d1/(d2+1e-10)

    is_discr = r<ratio

    ix2 = ix2[is_discr]

    ix1 = np.arange(D.shape[0])[is_discr]



    matches = np.stack((ix1,ix2), axis=-1)

    return matches


def computeH(p1, p2):

    '''

    INPUTS:

        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  

                 coordinates between two images

    OUTPUTS:

     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 

            equation

    '''

    assert(p1.shape[1]==p2.shape[1])

    assert(p1.shape[0]==2)

    #############################

    # TO DO ...

    A = np.zeros((2*p1.shape[1],9))

    p1 = p1.T

    p2 = p2.T

    

    length = p1.shape[0]

    for i in range(0,length):

        u,v = p1[i,0], p1[i,1]

        x,y = p2[i,0], p2[i,1]

        A[i*2,:] = np.array([-x,-y,-1,0,0,0,x*u,y*u,u])

        A[i*2+1,:] = np.array([0,0,0,-x,-y,-1,v*x,v*y,v])



    [D,V] = np.linalg.eig(np.matmul(A.T,A))

    idx = np.argmin(D)

    H2to1 = np.reshape(V[:,idx], (3,3))

    return H2to1



def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):

    '''

    Returns the best homography by computing the best set of matches using

    RANSAC

    INPUTS

        locs1 and locs2 - matrices specifying point locations in each of the images

        matches - matrix specifying matches between these two sets of point locations

        nIter - number of iterations to run RANSAC

        tol - tolerance value for considering a point to be an inlier



    OUTPUTS

        bestH - homography matrix with the most inliers found during RANSAC

    ''' 

    ###########################

    # TO DO ...

    H2to1 = np.zeros((3,3))

    maxInliers = -1

    bestInliers = np.zeros((1,1))



    num_matches = matches.shape[0]

    p1 = locs1[matches[:,0], 0:2].T

    p2 = locs2[matches[:,1], 0:2].T



    for i in range(0, 1000):

        idx = np.random.choice(len(matches), 4)

        rand1 = p1[:,idx]
        rand2 = p2[:,idx]

        H = computeH(rand1, rand2)



        p2_est = np.append(p2.T, np.ones([len(p2.T),1]),1)

        p2_est = p2_est.T

        p1_est = np.matmul(H,p2_est)

        p1_est = p1_est/p1_est[2,:]



        actual_diff = np.square(p1[0,:] - p1_est[0,:]) + np.square(p1[1,:] - p1_est[1,:])

        inliers = actual_diff < tol**2

        numInliers = sum(inliers)

        

        if numInliers > maxInliers:

            maxInliers = numInliers

            bestInliers = inliers



    H2to1 = computeH(p1[:,bestInliers], p2[:,bestInliers])

    return H2to1



def warpH(im, H2to1):

    print(im.shape)
    print(H2to1.shape)
    out_size = (1200,800)

    imWarped = cv2.warpPerspective(im, H2to1, out_size)

    imWarped = np.uint8(imWarped)

    homorgrahpy_file = '../results/q6_1.npy'

    np.save(homorgrahpy_file, H2to1)

    return imWarped



def imageStitching(im1, im2, H2to1,out_size):

    '''

    Returns a panorama of im1 and im2 using the given 

    homography matrix



    INPUT

        Warps img2 into img1 reference frame using the provided warpH() function

        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear

                 equation

    OUTPUT

        Blends img1 and warped img2 and outputs the panorama image

    '''

    #######################################

    # TO DO ...



    
    im1h, imw1 = im1.shape[:2]

    im2Warped = cv2.warpPerspective(im2, H2to1, out_size)

    im1Warped = cv2.warpPerspective(im1, np.identity(3), out_size)



    mask1 = distance_transform_edt(im1Warped)

    mask2 = distance_transform_edt(im2Warped)



    result1 = np.multiply(im1Warped,mask1)

    result2 = np.multiply(im2Warped,mask2)



    pano_im = np.divide(np.add(result1, result2), np.add(mask1, mask2))

    pano_im = np.nan_to_num(pano_im)

    pano_im = np.uint8(pano_im)

    return pano_im





def imageStitching_noClip(im1, im2, H2to1):

    '''

    Returns a panorama of im1 and im2 using the given 

    homography matrix without cliping.

    ''' 

    ######################################

    # TO DO ...



    imh1, imw1, imd1 = im1.shape

    imh2, imw2, imd2 = im2.shape



    corners = np.array([[0,imw2,0,imw2],[0,0,imh1,imh1],[1,1,1,1]])

    warpedCorners = np.matmul(H2to1, corners)

    warpedCorners = warpedCorners/warpedCorners[2,:]

    warp_corner = np.ceil(warpedCorners)



    row1 = im1.shape[0]

    col1 = im2.shape[1]

    maxrow = max(row1,max(warp_corner[1,:]))

    minrow = min(1,min(warp_corner[1,:]))

    maxcol = max(col1,max(warp_corner[0,:]))

    mincol = min(1,min(warp_corner[0,:]))



    scale = (maxcol-mincol)/(maxrow-minrow)



    W_out = 2000

    height = 1000

    out_size = (W_out, int(round(height)+1))



    s = W_out / (maxcol-mincol)

    scaleM = np.array([[s,0,0],[0,s,0],[0,0,1]])

    transM = np.array([[1,0,0],[0,1,-minrow],[0,0,1]])

    M = np.matmul(scaleM,transM)



    im2Warped = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)

    im1Warped = cv2.warpPerspective(im1, np.matmul(scaleM,transM), out_size)



    mask1 = distance_transform_edt(im1Warped)

    mask2 = distance_transform_edt(im2Warped)



    result1 = np.multiply(im1Warped,mask1)

    result2 = np.multiply(im2Warped,mask2)



    pano_im = np.divide(np.add(result1, result2), np.add(mask1, mask2))

    pano_im = np.nan_to_num(pano_im)

    pano_im = np.uint8(pano_im)

    return pano_im



def generatePanaroma(img1,img2):

    im1 = cv2.imread(img1)

    im2 = cv2.imread(img2)

    locs1, desc1 = briefLite(im1)

    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)

    Homo = ransacH(matches, locs1, locs2, num_iter=10000, tol=2)
    
    return Homo
#imageStitching_noClip(im1,im2,H2to1)

def process(folder,img1,img2,img3):
    
    im1 = cv2.imread(folder+img1)
    
    im2 = cv2.imread(folder+img2)
    
    im3 = cv2.imread(folder+img3)
    
    Homo12 = generatePanaroma(folder+img1,folder+img2)
    
    Homo23 = generatePanaroma(folder+img2,folder+img3)
    
    Homo13 = np.matmul(Homo12,Homo23)
    
    warped = warpH(im1, Homo13)
    
    out_size = (600,600)
    
    pano23 = imageStitching_noClip(im2,im3,Homo23)
    
    out_size = (800,600)
    
    panorama = imageStitching_noClip(warped,pano23,Homo13)
    
    cv2.imwrite(folder+'Panorama.jpg',panorama)
    
    
def process2(folder,img1,img2):
    im1 = cv2.imread(folder+img1)
    im2 = cv2.imread(folder+img2)
    Homo12 = generatePanaroma(folder+img1,folder+img2)
    panorama = imageStitching_noClip(im1,im2,Homo12)
    cv2.imwrite(folder+'Panorama.jpg',panorama)
    
def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument('string', type=str, default="./ubdata/",
                        help="Resources folder,i.e, folder in which images are stored")
    args = parser.parse_args()
    return args   

def getImages(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") and not filename ==("Panorama.jpg"):
            images.append(filename)
    images.sort()
    return images

if __name__ == '__main__':
    #args = parse_args()
    #folder = args.string
    folder="./data/"
    Images = getImages(folder)
    print(Images)
    im1 = cv2.imread(folder+Images[0])
    print(folder+Images[0])
    print(im1)
    if len(Images)==3:
         process(folder,Images[0],Images[1],Images[2])
    if(len(Images)==2):
        process2(folder,Images[0],Images[1])
    print("success")
