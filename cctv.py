import cv2



cap = cv2.VideoCapture(1)
cascade_filename = '../haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    if frame is not None:
        # Display the resulting frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20) )
        for box in results:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
            if len(results) != 0:
                cv2.putText(frame, "detected", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
            
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
