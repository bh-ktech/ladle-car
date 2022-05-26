import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model.conf = 0.3
model.classes = [0, 1]

camera = cv2.VideoCapture("rtsp://169.254.118.208/stream1")


def overlap(rect1, rect2):
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[1] > rect2[3] or rect1[3] < rect2[1])


while True:
    ret, img = camera.read()
    if not ret:
        while cv2.waitKey(100) == -1:
            pass
        break

    results = model(img)

    for *bb, conf, cls in results.xyxy[0]:

        s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)
        c1 = (255, 255, 0)
        c2 = (128, 0, 0)
        c3 = (0, 255, 0)

        # bbox
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color=c1, thickness=2)

        # class box
        cv2.rectangle(img, (int(bb[0]), int(bb[1])-20), (int(bb[0])+len(s)*10, int(bb[1])), c1, -1)

        # class name
        cv2.putText(img, s, (int(bb[0]), int(bb[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, c2, 1, cv2.LINE_AA)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
