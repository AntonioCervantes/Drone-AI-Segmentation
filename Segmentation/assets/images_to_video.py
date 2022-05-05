import numpy as np
import cv2
import tensorflow as tf

if __name__ == "__main__":
    """ Video Path """
    video_path = "../data/drone_vids/clip1/clip1.mp4"


    """ Reading frames """
    H = 320
    W = 480
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output.avi', fourcc, 10, (W, H), True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            out.release()
            break

        
        H = 320
        W = 480
        
        frame = cv2.resize(frame, (W, H))
        original_frame = frame


        # My code
        frame = frame / 255.0
        frame = frame.astype(np.float32)
        mask = model.predict(np.expand_dims(frame, axis=0))[0]

        mask = np.argmax(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        num_classes = 23
        mask = mask * (255/num_classes)
        mask = mask.astype(np.int32)
        mask = np.concatenate([mask, mask, mask], axis=2)

        #combine_frame = original_frame * mask
        #combine_frame = combine_frame.astype(np.uint8)

        cv2.imwrite(f"../data/drone_vids/clip1/{idx}.png", mask)
        idx += 1

        out.write(mask)