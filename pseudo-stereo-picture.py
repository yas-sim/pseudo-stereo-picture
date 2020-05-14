import sys

import numpy as np 
import cv2

from openvino.inference_engine import IECore

def main():
    if len(sys.argv)<2:
        print('please specify an input file', file=sys.stderr)
        sys.exit(-1)
    
    input_img = cv2.imread(sys.argv[1])

    ie = IECore()

    model = 'midasnet'  # image [1, 3, 384, 384] inverse_depth [1, 384, 384]
    model = './public/'+model+'/FP16/'+model
    net_depth = ie.read_network(model+'.xml', model+'.bin')
    inBlobName_depth = list(net_depth.inputs.keys())[0]
    outBlobName_depth = list(net_depth.outputs.keys())[0]
    inBlobShape_depth = net_depth.inputs[inBlobName_depth].shape
    outBlobShape_depth = net_depth.outputs[outBlobName_depth].shape
    #print(inBlobName_depth, inBlobShape_depth, outBlobName_depth, outBlobShape_depth)
    execNet_depth = ie.load_network(net_depth, 'CPU')

    model = 'gmcnn-places2-tf'  # Placeholder [1, 3, 512, 680] Placeholder_1 [1, 1, 512, 680] Minimum [1, 3, 512, 680]
    model = './public/'+model+'/FP16/'+model
    net_inpaint = ie.read_network(model+'.xml', model+'.bin')
    inBlobName_inpaint0 = list(net_inpaint.inputs.keys())[0]
    inBlobName_inpaint1 = list(net_inpaint.inputs.keys())[1]
    outBlobName_inpaint = list(net_inpaint.outputs.keys())[0]
    inBlobShape_inpaint0 = net_inpaint.inputs[inBlobName_inpaint0].shape
    inBlobShape_inpaint1 = net_inpaint.inputs[inBlobName_inpaint1].shape
    outBlobShape_inpaint = net_inpaint.outputs[outBlobName_inpaint].shape
    #print(inBlobName_inpaint0, inBlobShape_inpaint0, inBlobName_inpaint1, inBlobShape_inpaint1, outBlobName_inpaint, outBlobShape_inpaint)
    execNet_inpaint = ie.load_network(net_inpaint, 'CPU')

    # Generate depth image
    img = cv2.resize(input_img, (inBlobShape_depth[3], inBlobShape_depth[2]))
    img = img.transpose((2,0,1))
    img = img.reshape(inBlobShape_depth)
    res_depth = execNet_depth.infer(inputs={inBlobName_depth: img})
    invDepth = res_depth[outBlobName_depth].transpose((1,2,0))
    resizedInvDepth = cv2.resize(invDepth, (inBlobShape_inpaint0[3], inBlobShape_inpaint0[2]))
    depth_min = resizedInvDepth.min()
    depth_max = resizedInvDepth.max()
    print('depth range:', depth_min, depth_max)

    resizedImg = cv2.resize(input_img, (inBlobShape_inpaint0[3], inBlobShape_inpaint0[2]))

    imageStack = []
    shiftRange = 20
    print('Shift range:', -shiftRange, shiftRange)
    for shift in range(-shiftRange, shiftRange+1, 2):
        print('Shift:', shift)
        img = np.full(inBlobShape_inpaint0, 255, dtype=np.uint8)
        msk = np.full(inBlobShape_inpaint1, -1.0, dtype=np.float32)

        for y in range(inBlobShape_inpaint0[2]):
            for x in range(inBlobShape_inpaint0[3]):
                pixel = resizedImg[y,x,:]
                depth = resizedInvDepth[y,x]
                xx = int(x + (shift*((depth-depth_min)/depth_max)))  # Shift pixel position by depth
                # Check depth (Z buffer) and X coordinate before fill pixel
                if msk[0,0,y,x]<depth and xx>=0 and xx<inBlobShape_inpaint0[3]:
                    img[0,:,y,xx] = pixel
                    msk[0,0,y,xx] = depth

        msk = np.where(msk==-1., 1.0, 0.0)   # non filled pixels = lacking(=1.0)

        res_inpaint = execNet_inpaint.infer(inputs={inBlobName_inpaint0: img, inBlobName_inpaint1: msk})

        out = np.transpose(res_inpaint[outBlobName_inpaint], (0, 2, 3, 1)).astype(np.uint8)
        out = cv2.cvtColor(out[0], cv2.COLOR_RGB2BGR)

        imageStack.append(out)    # Append the new picture to stack
        cv2.imshow('Result', out)

        key = cv2.waitKey(1)
        if key==27:
            break

    sbs_img = np.hstack((imageStack[0], imageStack[-1]))
    cv2.imshow('Side-by-side stereo image', sbs_img)
    print('Output the side-by-side stereo image to \'sbs3d.jpg\'')
    cv2.imwrite('sbs3d.jpg', sbs_img)

    # Show results
    while True:
        for image in imageStack:
            cv2.imshow('Result', image)
            if cv2.waitKey(30)==27: return
        for image in imageStack[::-1]:
            cv2.imshow('Result', image)
            if cv2.waitKey(30)==27: return

if __name__ == '__main__':
    main()
