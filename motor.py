from periphery import GPIO, PWM
import time
import threading, ctypes
import logging

# ================= 用户配置区域 =================
MOTOR_GPIO_CONFIG = [
    {"name": "FL", "in2":36 ,"in1":39 ,"phase_A":40 ,"phase_B":42 },
    {"name": "FR", "in2":47 ,"in1":63 ,"phase_A":96 ,"phase_B":114 },
    {"name": "BL", "in2":34 ,"in1":44 ,"phase_A":45 ,"phase_B":46 },
    {"name": "BR", "in2":35 ,"in1":101 ,"phase_A":100 ,"phase_B":99 },
]

MOTOR_PWM_CONFIG = [
    {"chip":0 ,"channel":0 },    # FL,MotorA, fd8b0000, 15
    {"chip":1 ,"channel":0 },    # FR,MotorB, fd8b0010, 16
    {"chip":5 ,"channel":0 },    # BL,MotorC, febf0020, 62
    {"chip":4 ,"channel":0 },    # BR,MotorD, febe0030, 97
]

# PID 参数
Kp = 1.8
Ki = 0.004
Kd = 0.002

# 麦轮最大速度限制 (防止归一化前溢出)
MECANUM_MAX_SPEED = 400  # RPM 
# ===========================================

logger = logging.getLogger(__name__)

def get_gpio_chip_and_line(global_gpio_num):
    """
    将全局 GPIO 编号转换为 (Chip Path, Line Offset)
    规则: 每 32 个 GPIO 为一个 chip
    """
    chip_index = global_gpio_num // 32
    line_offset = global_gpio_num % 32
    chip_path = f"/dev/gpiochip{chip_index}"
    return chip_path, line_offset

def get_tid():
    return ctypes.CDLL('libc.so.6').syscall(186)

class EncoderCounter:
    def __init__(self, global_pin_a, global_pin_b):
        self.count = 0
        self.last_state = 0
        self.lock = threading.Lock()
        self.running = True
        self.gpio_a = None
        self.gpio_b = None
        self.table = [
            [  0, -1, +1,  0 ],  # last 00
            [ +1,  0,  0, -1 ],  # last 01
            [ -1,  0,  0, +1 ],  # last 10
            [  0, +1, -1,  0 ]   # last 11
        ]
        
        chip_path_a, line_a = get_gpio_chip_and_line(global_pin_a)
        chip_path_b, line_b = get_gpio_chip_and_line(global_pin_b)
        
        try:
            self.gpio_a = GPIO(chip_path_a, line_a, "in", edge="both", bias="pull_up")
            self.gpio_b = GPIO(chip_path_b, line_b, "in", edge="both", bias="pull_up")

            self.curr_a = 1 if self.gpio_a.read() else 0
            self.curr_b = 1 if self.gpio_b.read() else 0
            self.last_state = self.curr_a << 1 | self.curr_b
            
            self.thread = threading.Thread(target=self._poll_events)
            self.thread.daemon = True
            self.thread.start()
        except Exception as e:
            raise RuntimeError(f"Failed to init Encoder (A:{global_pin_a}, B:{global_pin_b}): {e}")

    def _poll_events(self):
        logger.info(f"Encoder thread started for GPIO A:{self.gpio_a.line}, B:{self.gpio_b.line}")
        # try:
        #     import os
        #     tid = get_tid()
        #     os.sched_setscheduler(tid, os.SCHED_RR, os.sched_param(10))
        # except Exception as e:
        #     logger.warning(f"Failed to set real-time scheduling for encoder thread: {e}")

        gpios = [self.gpio_a, self.gpio_b]
        dbg_events_a = 0
        dbg_events_b = 0
        while self.running:
            triggered_gpios = GPIO.poll_multiple(gpios, 0.001)
            if not triggered_gpios:
                continue

            for gpio in triggered_gpios:
                try:
                    event = gpio.read_event()
                    if gpio == self.gpio_a:
                        dbg_events_a += 1
                        if event.edge == "falling":
                            self.curr_a = 0
                        else:
                            self.curr_a = 1
                    else:
                        dbg_events_b += 1
                        if event.edge == "falling":
                            self.curr_b = 0
                        else:
                            self.curr_b = 1
                except Exception as e:
                    logger.error(f"read event : {e}")
                    continue

            current_state = (self.curr_a << 1) | self.curr_b
            if current_state != self.last_state:
                delta = self.table[self.last_state][current_state]
                with self.lock:
                    self.count += delta
                self.last_state = current_state

            # total_events = dbg_events_a + dbg_events_b
            # if total_events % 100 == 0 and total_events > 0:
            #     logger.debug(f"Encoder GPIO A:{self.gpio_a.line} events={dbg_events_a}, B:{self.gpio_b.line} events={dbg_events_b}, ratio={dbg_events_a/total_events:.2f}/{dbg_events_b/total_events:.2f}")

    def get_speed(self, interval):
        with self.lock:
            val = self.count
            self.count = 0
        # logger.debug(f"{self.gpio_a.line}:{self.gpio_b.line}, get_speed: count={val}, interval={interval:.3f}s")
        return val / interval

    def cleanup(self):
        self.running = False
        if self.gpio_a: self.gpio_a.close()
        if self.gpio_b: self.gpio_b.close()

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp, self.ki, self.kd, self.setpoint = kp, ki, kd, setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.alpha = 0.01  # 微分滤波器系数，0 < alpha <= 1.0
        self.filtered_derivative = 0.0

    def compute(self, current_value):
        now = time.time()
        dt = max(0.001, now - self.last_time)
        error = self.setpoint - current_value

        P = self.kp * error
        self.integral = max(-100, min(100, self.integral + error * dt))
        I = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        # 使用一阶低通滤波器平滑微分项
        # 公式: y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
        self.filtered_derivative = self.alpha * derivative + (1 - self.alpha) * self.filtered_derivative
        D = self.kd * self.filtered_derivative
        # logger.debug(f"PID compute: setpoint={self.setpoint:.1f}, current={current_value:.1f}, error={error:.1f}, P={P:.1f}, I={I:.1f}, D={D:.1f}")

        self.prev_error = error
        self.last_time = now
        return P + I + D

class Motor:
    def __init__(self, name, gpio_cfg, pwm_cfg, direction_inverted=False):
        self.name = name
        self.CPR = 880  # 20减速比 * 11线数 * 4(四倍频)
        self.DEADZONE_MIN = 5.0 # 死区 (%)
        self.PWM_FREQ = 20000     # Hz
        self.pid = PIDController(Kp, Ki, Kd, 0)
        self.in1_gpio = None
        self.in2_gpio = None
        self.pwm = None
        self.direction = -1 if direction_inverted else 1
        
        # 1. 初始化方向引脚 (IN1, IN2)
        try:
            chip_in2, line_in2 = get_gpio_chip_and_line(gpio_cfg["in2"])
            chip_in1, line_in1 = get_gpio_chip_and_line(gpio_cfg["in1"])
            
            self.in2_gpio = GPIO(chip_in2, line_in2, "out")
            self.in1_gpio = GPIO(chip_in1, line_in1, "out")
            
            self.in2_gpio.write(False)
            self.in1_gpio.write(False)
            
            logger.info(f"  -> DIR Initialized: IN2(Global:{gpio_cfg['in2']}->{chip_in2}:{line_in2}), IN1(Global:{gpio_cfg['in1']}->{chip_in1}:{line_in1})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to init DIR GPIOs for {name}: {e}")

        # 2. 初始化编码器
        try:
            self.encoder = EncoderCounter(gpio_cfg["phase_A"], gpio_cfg["phase_B"])
            chip_a, line_a = get_gpio_chip_and_line(gpio_cfg["phase_A"])
            logger.info(f"  -> Encoder Initialized: A(Global:{gpio_cfg['phase_A']}->{chip_a}:{line_a})")
        except Exception as e:
            raise RuntimeError(f"Failed to init Encoder for {name}: {e}")

        # 3. 初始化 PWM
        try:
            self.pwm = PWM(pwm_cfg['chip'], pwm_cfg['channel'])
            self.pwm.frequency = self.PWM_FREQ
            self.pwm.duty_cycle = 0
            self.pwm.polarity = "normal"
            self.pwm.enable()
            logger.info(f"  -> PWM Initialized: Chip{pwm_cfg['chip']}, Ch{pwm_cfg['channel']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to init PWM for {name}: {e}")

    def set_speed(self, target_speed):
        self.pid.setpoint = target_speed

    def update(self, dt):
        speed = self.encoder.get_speed(dt)    # 脉冲/秒
        speed = self.direction * speed * 60 / self.CPR  # 转每分钟 (RPM)

        # 避免进入死区效应
        if self.pid.setpoint == 0 and abs(speed) < self.DEADZONE_MIN:
            self._drive(0, 1)
            return speed, 0
        
        output = self.pid.compute(speed)
        duty = abs(output)
        direction = 1 if output >= 0 else -1
        
        duty = max(0, min(100, duty))
        if 0 < duty < self.DEADZONE_MIN:
            duty = self.DEADZONE_MIN

        logger.debug(f"{self.name}, target={self.pid.setpoint:.1f}RPM, actual={speed:.1f}RPM, output={output:.1f}, duty={duty:.1f}%, direction={direction}")
        self._drive(duty, direction)
        return speed, duty

    def _drive(self, duty_percent, direction):
        if direction == 1:
            self.in1_gpio.write(True)
            self.in2_gpio.write(False)
        else:
            self.in1_gpio.write(False)
            self.in2_gpio.write(True)
            
        if self.pwm:
            self.pwm.duty_cycle = duty_percent / 100.0

    def cleanup(self):
        if self.in1_gpio: self.in1_gpio.close()
        if self.in2_gpio: self.in2_gpio.close()
        if self.encoder: self.encoder.cleanup()
        if self.pwm:
            self.pwm.duty_cycle = 0
            self.pwm.disable()
            self.pwm.close()

class MecanumController:
    def __init__(self, motors_dict):
        """
        motors_dict: {'FL': Motor, 'FR': Motor, 'BL': Motor, 'BR': Motor}
        """
        self.motors = motors_dict
        self.max_speed = MECANUM_MAX_SPEED
        self.last_time = time.perf_counter()
        self.vx = 0
        self.vy = 0
        self.vw = 0
        self.LOOP_TIME = 0.001
        self.running = True
        self.thread = threading.Thread(target=self._control_loop)
        self.thread.daemon = True
        self.thread.start()

    def _control_loop(self):
        """
        :param vx: 前后速度 (正=前进)
        :param vy: 左右速度 (正=向右平移)
        :param vw: 旋转速度 (正=逆时针旋转)
        """
        while self.running:
            now = time.perf_counter()
            dt = max(0.001, now - self.last_time)
            self.last_time = now

            # 麦轮运动学解算 (标准布局：左前/右后辊子朝内)
            # 如果发现方向反了，请调整这里的 +/-
            v_fl = self.vx - self.vy - self.vw
            v_fr = self.vx + self.vy + self.vw
            v_bl = self.vx + self.vy - self.vw 
            v_br = self.vx - self.vy + self.vw 
            
            # 归一化
            max_val = max(abs(v_fl), abs(v_fr), abs(v_bl), abs(v_br))
            
            if max_val > self.max_speed:
                scale = self.max_speed / max_val
                v_fl *= scale
                v_fr *= scale
                v_bl *= scale
                v_br *= scale
                
            # 下发目标速度给每个电机的 PID
            self.motors['FL'].set_speed(v_fl)
            self.motors['FR'].set_speed(v_fr)
            self.motors['BL'].set_speed(v_bl)
            self.motors['BR'].set_speed(v_br)
            self.motors['FL'].update(dt)
            self.motors['FR'].update(dt) 
            self.motors['BL'].update(dt)
            self.motors['BR'].update(dt)
            time.sleep(self.LOOP_TIME)

    def stop(self):
        self.vx = 0
        self.vy = 0
        self.vw = 0

    def cleanup(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)

def main():
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s %(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', filename='motor.log', filemode='w')
    logger.info("=== 麦轮全向运动测试 (Mecanum Omni-Directional Test) ===")
    logger.info("Initializing Motors...")
    
    if len(MOTOR_GPIO_CONFIG) != len(MOTOR_PWM_CONFIG):
        logger.error("Config lists length mismatch!")
        return
    
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
            logger.error(f"Motor {gpio_cfg['name']}: {e}")
    
    if len(motors_list) < 4:
        logger.error("Not enough motors initialized. Exiting.")
        return
    
    controller = MecanumController(motors_dict)
    
    # 测试参数
    SPEED_FWD = 60      # 前进/后退速度 (RPM)
    SPEED_STRAFE = 60    # 平移速度 (RPM)
    SPEED_ROTATE = -30    # 旋转速度 (RPM)
    DURATION = 3.0        # 每个动作持续时间 (秒)
    
    logger.info(f"\n开始测试序列 (每个动作持续 {DURATION} 秒)...")
    logger.info("请观察小车动作是否符合预期。\n")
    action = input("f,b,r,l,r1,r2,fr,fr1: ").strip().lower()
    
    match action:
        case 'f':
            # --- 测试 1: 前进 ---
            logger.info("[1] 前进 (Forward)")
            controller.vx = SPEED_FWD
            controller.vy = 0
            controller.vw = 0
            
        case 'b':
            # --- 测试 2: 后退 ---
            logger.info("[2] 后退 (Backward)")
            controller.vx = -SPEED_FWD
            controller.vy = 0
            controller.vw = 0
            
        case 'r':
            # --- 测试 3: 向右平移 ---
            logger.info("[3] 向右平移 (Strafe Right)")
            controller.vx = 0
            controller.vy = SPEED_STRAFE
            controller.vw = 0

        case 'l':
            # --- 测试 4: 向左平移 ---
            logger.info("[4] 向左平移 (Strafe Left)")
            controller.vx = 0
            controller.vy = -SPEED_STRAFE
            controller.vw = 0

        case 'r1':
            # --- 测试 5: 原地逆时针旋转 ---
            logger.info("[5] 原地逆时针旋转 (Rotate CCW)")
            controller.vx = 0
            controller.vy = 0
            controller.vw = SPEED_ROTATE
            
        case 'r2':
            # --- 测试 6: 原地顺时针旋转 ---
            logger.info("[6] 原地顺时针旋转 (Rotate CW)")
            controller.vx = 0
            controller.vy = 0
            controller.vw = -SPEED_ROTATE

        case 'fr':
            # --- 测试 7: 斜向移动 (右前) ---
            logger.info("[7] 右前方斜移 (Diagonal Front-Right)")
            controller.vx = SPEED_FWD
            controller.vy = SPEED_STRAFE
            controller.vw = 0

        case 'fr1':
            # --- 测试 8: 复合运动 (前进 + 旋转) ---
            logger.info("[8] 前进同时逆时针旋转 (Forward + Rotate)")
            controller.vx = SPEED_FWD
            controller.vy = 0
            controller.vw = SPEED_ROTATE

        case _:
            pass

    time.sleep(DURATION)
    logger.info("\n=== 测试完成，停车 ===")
    controller.stop()
    time.sleep(1)

    logger.info("Cleaning up motors...")
    controller.cleanup()
    for m in motors_list:
        m.cleanup()
    logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()
