import cv2
import torch
import timeit

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.2
model.classes = [0, 1]

camera = cv2.VideoCapture("rtsp://169.254.28.68/stream1")


def overlap(bb1, bb2):
    return not (bb1[0] > bb2[2] or bb1[1] > bb2[3] or bb1[2] < bb2[0] or bb1[3] < bb2[1])


while True:
    ret, img = camera.read()
    if not ret:
        while cv2.waitKey(100) == -1:
            pass
        break
    start_t = timeit.default_timer()
    results = model(img)
    car = []
    ladle = []
    for *bb, conf, cls in results.xyxy[0]:

        s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)
        c1 = (255, 255, 0)
        c2 = (128, 0, 0)
        c3 = (0, 255, 0)
        # # bbox
        # cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color=cc, thickness=2)
        #
        # # class box
        # cv2.rectangle(img, (int(bb[0]), int(bb[1])-20), (int(bb[0])+len(s)*10, int(bb[1])), cc, -1)
        #
        # # class name
        # cv2.putText(img, s, (int(bb[0]), int(bb[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)
        #
        # print(posx)
        if model.names[int(cls)] == 'Car':
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color=c3, thickness=2)
            cv2.rectangle(img, (int(bb[0]), int(bb[1]) - 20), (int(bb[0]) + len(s) * 10, int(bb[1])), c3, -1)
            cv2.putText(img, s, (int(bb[0]), int(bb[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, c2, 1, cv2.LINE_AA)

            for i in range(4):
                car.append(int(bb[i]))

        elif model.names[int(cls)] == 'Ladle':
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color=c1, thickness=2)
            cv2.rectangle(img, (int(bb[0]), int(bb[1]) - 20), (int(bb[0]) + len(s) * 10, int(bb[1])), c1, -1)
            cv2.putText(img, s, (int(bb[0]), int(bb[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, c2, 1, cv2.LINE_AA)

            for i in range(4):
                ladle.append(int(bb[i]))

        if len(car) and len(ladle) > 0:
            if overlap(car, ladle) is True:
                cv2.putText(img, "warning", (150, 250), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 3, cv2.LINE_AA)

    end_t = timeit.default_timer()
    FPS = f'FPS : {str(int(1. / (end_t - start_t)))}'
    cv2.putText(img, FPS, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
