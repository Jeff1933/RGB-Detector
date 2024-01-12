import cv2
import numpy as np

def adjust_sv_thresholds(hsv_image, lower_hsv, upper_hsv, delta_s=35, delta_v=35):
    # 计算HSV图像的V通道的平均值
    avg_brightness = np.mean(hsv_image[:, :, 2])
    
    # 根据平均亮度调整S和V的阈值
    if avg_brightness < 128:  # 如果图像偏暗
        lower_hsv[1] = max(lower_hsv[1] - delta_s, 0)  # 增加饱和度下限
        lower_hsv[2] = max(lower_hsv[2] - delta_v, 0)  # 增加亮度下限
    else:  # 如果图像偏亮
        lower_hsv[1] = min(lower_hsv[1] + delta_s, 255)  # 减少饱和度下限
        lower_hsv[2] = min(lower_hsv[2] + delta_v, 255)  # 减少亮度下限

    return lower_hsv, upper_hsv

def main():
    initial_lower = np.array([50, 70, 50])
    initial_upper = np.array([70, 255, 255])
    # 定义颜色的HSV范围
    lower = initial_lower.copy()
    upper = initial_upper.copy()
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            break
        # 将图像从BGR转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = adjust_sv_thresholds(hsv, lower, upper)
        print(f"Lower HSV: {lower}, Upper HSV: {upper}")
        # 识别颜色
        mask = cv2.inRange(hsv, lower, upper)

        # 查找轮廓
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 如果轮廓的面积大于一定值，标注物体并计算Hu矩
            if cv2.contourArea(contour) > 5000:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)
                #print("Hu Moments: ", hu_moments)

        # 显示图像
        cv2.imshow("Image", frame)

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 重置HSV范围以避免累积变化
        lower = initial_lower.copy()
        upper = initial_upper.copy()
            
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
