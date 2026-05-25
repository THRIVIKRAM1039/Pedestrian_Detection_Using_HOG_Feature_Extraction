import cv2
import imutils
import numpy as np
import os

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def non_max_suppression(boxes, overlapThresh=0.5):
    """Apply Non-Maximum Suppression to remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def detect_pedestrians(image_path, conf_threshold=0.5, min_w=30, min_h=60):
    """Detect pedestrians in a single image and return result info."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None, 0

    orig = image.copy()

    # Resize preserving aspect ratio (max width 800 to retain more detail than 400)
    image = imutils.resize(image, width=min(800, image.shape[1]))

    # Detect pedestrians with finer scale search
    (regions, weights) = hog.detectMultiScale(
        image,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.02,
        hitThreshold=0
    )

    # Fallback: if no detections on resized image, try original size
    if len(regions) == 0:
        (regions, weights) = hog.detectMultiScale(
            orig,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.02,
            hitThreshold=0
        )
        image = orig

    # Filter by confidence (SVM decision score) and minimum box size
    filtered = []
    if len(regions) > 0:
        for i, (x, y, w, h) in enumerate(regions):
            conf = float(weights[i]) if i < len(weights) else 0.0
            if conf >= conf_threshold and w >= min_w and h >= min_h:
                filtered.append([x, y, w, h])

        if len(filtered) > 0:
            filtered = np.array(filtered)
            filtered = non_max_suppression(filtered, overlapThresh=0.5)
        else:
            filtered = []

    # Draw bounding boxes
    if len(filtered) > 0:
        for (x, y, w, h) in filtered:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, len(filtered)


# Process all test images
image_files = ['no pedes.jpeg', 'pedes no 1.jpeg', 'pedes.jpg']

for img_file in image_files:
    if os.path.exists(img_file):
        result_img, count = detect_pedestrians(img_file)
        if result_img is not None:
            # Save result
            output_name = f"result_{img_file}"
            cv2.imwrite(output_name, result_img)
            status = f"Pedestrian(s) Detected: {count}" if count > 0 else "No Pedestrian Detected"
            print(f"[{img_file}] {status} -> Saved: {output_name}")
    else:
        print(f"[SKIP] File not found: {img_file}")

print("\nAll images processed. Check the result_*.jpeg files in the folder.")

