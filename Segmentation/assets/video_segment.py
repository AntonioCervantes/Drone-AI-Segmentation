import numpy as np
import cv2
import tensorflow as tf

if __name__ == "__main__":
    """ Video Path """
    video_path = "../data/drone_vids/clip1/clip1.mp4"

    """ Load the model """
    model = tf.keras.models.load_model("../results/models/model_11.h5")

    """ Reading frames """
    #vs = cv2.VideoCapture(video_path)
    #_, frame = vs.read()
    #H, W, _ = frame.shape
    #vs.release()
    H = 320
    W = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../data/drone_vids/clip1/output.mp4', fourcc, 10, (W, H), True)

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
        mask = mask.astype(np.uint8)

        #combine_frame = original_frame * mask
        #combine_frame = combine_frame.astype(np.uint8)
        #combine_frame = mask
        #combine_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        #combine_frame = combine_frame.astype(np.uint8)

        # Write Predictions
        cv2.imwrite(f"../data/drone_vids/clip1/img/pred/{idx}.png", mask)
        cv2.imwrite(f"../data/drone_vids/clip1/img/orig/original_{idx}.png", original_frame)
        idx += 1

        #out.write(mask)