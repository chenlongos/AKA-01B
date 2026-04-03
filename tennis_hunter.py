import cv2
import os, logging, time
import numpy as np
from motor import MecanumController, MOTOR_GPIO_CONFIG, MOTOR_PWM_CONFIG, Motor
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HARDWARE_MODE = 'rk3588'    # cpu, rk3588

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='th.log', filemode='w'
)

if HARDWARE_MODE == 'cpu':
    import onnxruntime as ort
elif HARDWARE_MODE == 'rk3588':
    from rknn.api import RKNN
    rknn = RKNN()
else:
    raise ValueError(f"不支持的硬件模式: {HARDWARE_MODE}")

# 初始化模型
if HARDWARE_MODE == 'cpu':
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.onnx')
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
elif HARDWARE_MODE == 'rk3588':
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.rknn')
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(target='rk3588')

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    YOLOv8 官方预处理函数，保持宽高比 resize + center pad
    """
    shape = img.shape[:2]  # current shape [H, W]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def yolo_infer(img_or_frame):
    img_size = 640

    if isinstance(img_or_frame, str):
        # 输入是图像路径
        orig_img = cv2.imread(img_or_frame)
        if orig_img is None:
            logging.warning(f" 无法读取图像: {img_or_frame}")
            return []
    else:
        # 输入是视频帧（ndarray）
        orig_img = img_or_frame

    H, W = orig_img.shape[:2]
    input_img = letterbox(orig_img, new_shape=(img_size, img_size))

    if HARDWARE_MODE == 'cpu':
        blob = cv2.dnn.blobFromImage(input_img, scalefactor=1 / 255.0, size=(img_size, img_size), swapRB=True, crop=False)
        outputs = session.run(None, {input_name: blob})
        pred = outputs[0].squeeze().T  # [C, N] -> [N, C]
    elif HARDWARE_MODE == 'rk3588':
        outputs = rknn.inference(inputs=[input_img])
        pred = outputs[0].squeeze().T  # [C, N] -> [N, C]

    boxes_xywh = pred[:, :4]  # cx, cy, w, h（YOLOv8 输出是 xywh 格式）
    conf_scores = pred[:, 4]
    mask = conf_scores > 0.25

    # dbg:
    # logging.debug(f"ONNX Output Shape: {pred.shape})
    # dbg_max_scores = np.max(pred[:, 4:], axis=1)
    # logging.debug(f"ONNX Max Scores Top 10: {np.sort(dbg_max_scores)[-10:]})
    # logging.debug(f"Raw output shape: {outputs[0].shape}")
    # logging.debug(f"Pred shape after processing: {pred.shape}")
    # logging.debug(f"Sample confidence: {conf_scores[:5]}")

    pred = pred[mask]
    boxes_xywh = boxes_xywh[mask]
    conf_scores = conf_scores[mask]

    boxes = []
    raw_boxes = []
    for i in range(len(boxes_xywh)):
        cx, cy, w, h = boxes_xywh[i]
        #  使用letterbox 的 pad 参数精确还原
        shape = orig_img.shape[:2]
        r = min(img_size / shape[0], img_size / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
        dw /= 2
        dh /= 2

        # 将 640x640 坐标还原到缩放后尺寸（new_unpad）
        x1 = (cx - w / 2 - dw) / r
        y1 = (cy - h / 2 - dh) / r
        x2 = (cx + w / 2 - dw) / r
        y2 = (cy + h / 2 - dh) / r

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        raw_boxes.append([x1, y1, x2, y2])

    # NMS
    raw_boxes = np.array(raw_boxes, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), 0.25, 0.45)

    if indices is not None and len(indices) > 0:
        for idx in indices:
            i = int(idx) if np.isscalar(idx) else int(idx[0])
            x1, y1, x2, y2 = raw_boxes[i]
            box = {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1)
            }
            boxes.append(box)

    return boxes

def release():
    if HARDWARE_MODE == 'rk3588':
        rknn.release()
    # 其它硬件无 release 操作

@dataclass
class Robot:
    FRAME_WIDTH = 640
    TENNIS_WIDTH_FAR = 160
    TENNIS_WIDTH_NEAR = 190
    MAX_SPEED = 400
    MIN_SPEED = MAX_SPEED // 40  
    status: str = "chase_tennis" # 机器人状态: chase_tennis, chase_bucket, grab_tennis, position_tennis, release_tennis
    box_cur_width: int = 0
    box_cur_height: int = 0
    box_cur_x: int = 0
    frame_height: int = 0

    idle_speed = 60 # 旋转时的线速度

    controller: MecanumController = field(init=False)

    def __post_init__(self):
        # 初始化车轮
        motors_list = []
        motors_dict = {}
        
        for i in range(len(MOTOR_GPIO_CONFIG)):
            gpio_cfg = MOTOR_GPIO_CONFIG[i]
            pwm_cfg = MOTOR_PWM_CONFIG[i]
            direction_inverted = (i % 2 == 0)  
            
            try:
                m = Motor(gpio_cfg["name"], gpio_cfg, pwm_cfg, direction_inverted=direction_inverted)
                motors_list.append(m)
                motors_dict[gpio_cfg["name"]] = m
            except Exception as e:
                logging.error(f"Motor {gpio_cfg['name']}: {e}")
        
        if len(motors_list) < 4:
            logging.error("Not enough motors initialized. Exiting.")
            return
        
        self.controller = MecanumController(motors_dict)
    
    def set_motor_speed(self, result):
        IMG_WIDTH = self.FRAME_WIDTH
        MAX_SPEED = self.MAX_SPEED
        MIN_SPEED = self.MIN_SPEED
        WHEEL_BASE = 10.0
        TARGET_X = IMG_WIDTH // 2
        TARGET_W = int(self.TENNIS_WIDTH_FAR * 0.6 + self.TENNIS_WIDTH_NEAR * 0.4)

        Kp_dist = 0.8
        Kp_angle = 0.02

        result_sorted = sorted(result, key=lambda x: x['w'], reverse=True)
        box = result_sorted[0]
        x, w, h = box["x"], box["w"], box["h"]
        self.box_cur_height = h
        self.box_cur_x = x
        self.box_cur_width = w
        logging.info("(box_cur_x, box_cur_width, box_cur_height) ==> %d, %d, %d", self.box_cur_x, self.box_cur_width, self.box_cur_height)

        if self.TENNIS_WIDTH_FAR < self.box_cur_width < self.TENNIS_WIDTH_NEAR:
            self.controller.vx = 0
            self.controller.vy = 0
            self.controller.vw = 0
            return 0, 0

        # 1. 计算偏差
        error_x = (x + w / 2) - TARGET_X
        error_w = w - TARGET_W

        # 2. 计算线性速度和角速度
        raw_v = -Kp_dist * error_w
        raw_omega = -Kp_angle * error_x

        # 3. 动态限速
        turn_factor = abs(error_x) / (IMG_WIDTH / 2)
        if turn_factor > 0.8:
            max_v = MIN_SPEED * 0.3
        else:
            max_v = MAX_SPEED

        v = max(min(raw_v, max_v), -max_v)

        if abs(v) < MIN_SPEED and abs(v) > 0:
            v = MIN_SPEED if v > 0 else -MIN_SPEED

        diff_speed = raw_omega * WHEEL_BASE

        self.controller.vx = v
        self.controller.vy = 0
        self.controller.vw = diff_speed

        return int(v), int(diff_speed)
    
    def idle(self):
        self.controller.vx = 0
        self.controller.vy = 0
        self.controller.vw = self.idle_speed

def main_v():
    all_timings = []    # 每帧的处理时间
    
    logging.info("正在打开摄像头...")
    cap = cv2.VideoCapture(6)
    if not cap.isOpened():
        raise IOError("无法打开摄像头")
    
    robot = Robot()

    while True:
        start_time = time.time() * 1000
        _ret, frame = cap.read()
        height, width = frame.shape[:2]
        robot.frame_height = height
        if width != robot.FRAME_WIDTH:
            logging.warning(f" 当前帧宽度 {width} 不等于 {robot.FRAME_WIDTH}，正在调整大小...")
            frame = cv2.resize(frame, (robot.FRAME_WIDTH, int(robot.FRAME_WIDTH * height / width)), interpolation=cv2.INTER_LINEAR)
            robot.frame_height = frame.shape[0]

        logging.debug(f" 解算开始")
        result = yolo_infer(frame)
        logging.debug(f" 解算完成")

        if result:
            logging.info(f"update_status: status ==> {robot.status}")
            v, diff_speed = robot.set_motor_speed(result)
            logging.info(f"set_motor_speed: (v, diff_speed) ==> {v}, {diff_speed}")
        else:
            logging.info("idle")
            robot.idle()
            continue
        
        all_timings.append(int(time.time() * 1000 - start_time))
        avg_time = int(sum(all_timings) / len(all_timings) if all_timings else 0)
        min_time = min(all_timings) if all_timings else 0
        max_time = max(all_timings) if all_timings else 0
        logging.info(f"当前帧处理时间: {all_timings[-1]} ms, 平均: {avg_time} ms, 最小: {min_time} ms, 最大: {max_time} ms")

if __name__ == "__main__":
    main_v()