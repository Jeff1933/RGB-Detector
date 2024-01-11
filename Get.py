import cv2
import numpy as np

# 定义目标物体的Hu矩  这是保龄球的
""" target_hu_moments = np.array([
    [3.34754646e-01],
    [8.23127846e-02],
    [4.93104332e-03],
    [3.10297390e-03],
    [1.21377001e-05],
    [8.90247941e-04],
    [-1.13216051e-08]
]) """
#这是手机的
target_hu_moments = np.array([
    [2.05278664e-01],
    [1.39372915e-02],
    [1.10817603e-05],
    [2.35175957e-07],
    [-1.15500271e-13],
    [2.03377352e-08],
    [3.61663767e-13]
])


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
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        lower_blue = np.array([100,70, 50])
        upper_blue = np.array([130, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([70, 255, 255])

        # 识别颜色
        maskr = detect_color(frame, lower_red, upper_red)
        maskg = detect_color(frame, lower_green, upper_green)   
        maskb = detect_color(frame, lower_blue, upper_blue)

        # 查找轮廓
        contoursb, _ = cv2.findContours(maskb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursr, _ = cv2.findContours(maskr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursg, _ = cv2.findContours(maskg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contours, color in zip([contoursb, contoursg, contoursr], ['Blue', 'Green', 'Red']):
            for contour in contours:
                # 如果轮廓的面积大于一定值，且颜色满足条件
                if cv2.contourArea(contour) > 5000:
                    moments = cv2.moments(contour)
                    hu_moments = cv2.HuMoments(moments)

                    # 比较前6个Hu矩是否与目标物体的一致
                    if np.allclose(hu_moments[:6], target_hu_moments[:6], atol=0.01):
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, f'Target {color}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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