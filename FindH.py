import cv2
import numpy as np

def detect_color(image, lower, upper):
    # 将图像从BGR转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 找出在颜色范围内的像素
    mask = cv2.inRange(hsv, lower, upper)

    return mask

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 定义颜色的HSV范围
        lower = np.array([100, 70, 50])
        upper = np.array([130, 255, 255])

        # 识别颜色
        mask = detect_color(frame, lower, upper)

        # 查找轮廓
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 如果轮廓的面积大于一定值，标注物体并计算Hu矩
            if cv2.contourArea(contour) > 5000:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)
                print("Hu Moments: ", hu_moments)

        # 显示图像
        cv2.imshow("Image", frame)

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()