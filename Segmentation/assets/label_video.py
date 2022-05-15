import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import os

DEBUG: bool = True
LOCAL: bool = True
min_size: int = 45

if __name__ == "__main__":
    """ Video Path """
    video_path = r"C:\Users\pedri\Desktop\ME 297-01_Deep_Learning\Aruco-classification\Segmentation\data\drone_vids\clip1\output1.mp4" if LOCAL else "../data/drone_vids/clip1/clip1.mp4"

    labels = ["paved-area",
        "dirt",
        "grass",
        "gravel",
        "water",
        "rocks",
        "pool",
        "vegetation",
        "roof",
        "wall",
        "window",
        "door",
        "fence",
        "fence-pole",
        "person",
        "dog",
        "car",
        "bicycle",
        "tree",
        "bald-tree",
        "ar-marker",
        "obstacle",
        "conflicting",
    ]
    div = 255 / len(labels)
    thresh = [(1 + div*i, (i + 1)*div) if i != 0 else (0, (i + 1)*div) for i in range(len(labels))]
    new_list = zip(thresh, labels)
    if DEBUG: print(thresh) 

    """ Reading frames """
    #vs = cv2.VideoCapture(video_path)
    #_, frame = vs.read()
    #H, W, _ = frame.shape
    #vs.release()
    H = 320
    W = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dir = r"C:\Users\pedri\Desktop\ME 297-01_Deep_Learning\Aruco-classification\Segmentation\data\drone_vids\clip1"
    out_name = "label1.mp4"
    
    out_path = os.path.join(out_dir, out_name)
    out = cv2.VideoWriter(out_path, fourcc, 10, (W, H), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break

        #H, W, _ = frame.shape
        H = 320
        W = 480
        
        y = cv2.resize(frame, (W, H))
        x = frame

        if x is None or y is None:
            print(f"Actual image Name: {frame}, Type: type{type(y)}")
            raise Exception("No image has been read! Check your path name.")


        thresh_tuple = [cv2.threshold(x, thresh[i][0], thresh[i][1], cv2.THRESH_BINARY) for i in range(len(thresh))]

        mask = [cv2.bitwise_and(x, x, mask = thresh_tuple[i][1]) for i in range(len(thresh_tuple))]
        mask = [cv2.erode(mask[i], None, iterations = 4) for i in range(len(mask))]
        mask = [cv2.dilate(mask[i], None, iterations = 4) for i in range(len(mask))]
        mask = [mask[i].astype(np.uint8) for i in range(len(mask))]
        #mask = [cv2.cvtColor(mask[i], cv2.COLOR_BGR2GRAY) for i in range(len(mask))]
        if DEBUG: print(type(mask[0]))
        if DEBUG: 
            cv2.imshow("Mask", mask[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for i in range(len(mask)):
            cnts = cv2.findContours(mask[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            cX = None
            cY = None

            if DEBUG: print(len(cnts))

            for c in cnts:
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                (cX, cY) = center

                
                if radius > min_size:
                    text = f"{labels[i]}"
                    z = cv2.putText(img = y, 
                                    text = text, 
                                    org = (cX, cY - 15),
                                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale = 0.5, 
                                    color = (255, 255, 255), 
                                    thickness = 2)
                    
        

            if 0:
                plt.figure()
                plt.imshow(x)

                plt.figure()
                plt.imshow(y)


                fig = plt.figure()
                plt.contourf(x)
                cbar = plt.colorbar(ticks = [int(thresh[i][1]) for i in range(len(thresh))], orientation = 'vertical')
                cbar.ax.set_yticklabels( [labels[i] for i in range(len(thresh))])#[labels[i] in range(0, len(thresh), 1)])
                
                if LOCAL:
                    color_dir = r'C:\Users\pedri\Desktop\ME 297-01_Deep_Learning\Aruco-classification\img\colorbar_min_size_60'
                    color_path = os.path.join(color_dir, f"colorbar_{pic_num}.png")
                else:
                    color_path = f"colorbar_{pic_num}.png"
                plt.savefig(color_path)


            #combine_frame = original_frame * mask
            #combine_frame = combine_frame.astype(np.uint8)
            #combine_frame = mask
            #combine_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            #combine_frame = combine_frame.astype(np.uint8)

            # Write Predictions
            # cv2.imwrite(f"../data/drone_vids/clip1/img/pred/{idx}.png", mask)
            # cv2.imwrite(f"../data/drone_vids/clip1/img/orig/original_{idx}.png", original_frame)
            out.write(z)

            idx += 1

