# AKA-01B
辰龙机器人(使用香橙派5Plus控制)

## 1. orangepi5plus系统设置

1. 使能pwm和gpio，用于控制电机
2. 安装conda和rknn-toolkit2环境，用于运行rknn模型
3. 配置udev规则，使用户有访问pwm和gpio的权限

## 2. 运行本项目

```bash
python tennis_hunter.py
```

## CHANGELOG

- 2026-04-03: 实现追网球功能