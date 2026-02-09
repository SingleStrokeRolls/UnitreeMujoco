import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config

import matplotlib.pyplot as plt
import numpy as np

import math

# plot
def plotting(t_store, gyro_store, acc_store, tof_store):

    # 4. 转 numpy
    t = np.array(t_store)
    gyro = np.array(gyro_store)
    acc  = np.array(acc_store)
    tof = np.array(tof_store)

    # print(t)

    # 从后往前找 t 最接近 0 的点（断点）
    idx = np.where(t <= 0.005)[0]          
    # print(idx)
    if idx.size:
        start = idx[-1]                  # 最后一次 reset 的起点
    else:
        start = 0                        # 没找到就全画
    # print(start)

    t   = t[start:] - t[start]           # 时间从 0 开始
    gyro = gyro[start:]
    acc  = acc[start:]
    tof = tof[start:]

    # print(acc)

    # 5. 画图
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    # ax[0].plot(t, gyro[:, 0], label='gx')
    # ax[0].plot(t, gyro[:, 1], label='gy')
    # ax[0].plot(t, gyro[:, 2], label='gz')
    # ax[0].set_ylabel('Angular velocity (rad/s)')
    # ax[0].legend()
    # ax[0].set_title("G1 IMU Data")

    # ax[1].plot(t, acc[:, 0], label='ax')
    # ax[1].plot(t, acc[:, 1], label='ay')
    # ax[1].plot(t, acc[:, 2], label='az')
    # ax[1].set_ylabel('Acceleration (m/s²)')
    # ax[1].set_yscale('symlog', linthresh=20)
    # ax[1].legend()

    magnitude = np.sqrt(np.sum(gyro**2, axis=1))
    ax[0].plot(t, magnitude, label='ang mag')
    ax[0].set_ylabel('Angu Mag(rad/s)')
    # ax[0].set_yscale('symlog', linthresh=20)
    ax[0].legend()

    magnitude = np.sqrt(np.sum(acc**2, axis=1))
    ax[1].plot(t, magnitude, label='acc mag')
    ax[1].set_ylabel('Acc Mag(m/s²)')
    ax[1].set_yscale('symlog', linthresh=20)
    ax[1].legend()

    ax[2].plot(t, tof, label='tof', color='k')
    ax[2].set_ylabel('Distance (m)')
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig("imu_plot.png")
    print("已保存 imu_plot.png")

def export_csv(t_store, gyro_store, acc_store, tof_store):

    # 4. 转 numpy
    t = np.array(t_store)
    gyro = np.array(gyro_store)
    acc  = np.array(acc_store)
    tof = np.array(tof_store)

    # print(t)

    # 从后往前找 t 最接近 0 的点（断点）
    idx = np.where(t <= 0.005)[0]          
    # print(idx)
    if idx.size:
        start = idx[-1]                  # 最后一次 reset 的起点
    else:
        start = 0                        # 没找到就全画
    # print(start)

    t   = t[start:] - t[start]           # 时间从 0 开始
    gyro = gyro[start:]
    acc  = acc[start:]
    tof = tof[start:]

    data = np.column_stack([t, acc, gyro, tof])

    # 保存为 CSV，带表头
    np.savetxt(
        'output.csv',
        data,
        delimiter=',',
        fmt='%.6f',           # 根据精度需求调整：%.3f, %.6f 等
        header='t,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,tof',
        comments=''           # 去掉 # 注释符号，让表头干净
    )

    print(f"CSV 已保存，共 {len(t)} 行数据")


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

# reset timestep
mj_model.opt.timestep = 0.002

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    # plot
    t_store   = []
    gyro_store = []   # 3 轴角速度
    acc_store  = []   # 3 轴加速度

    tof_store = []

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    fall = "sit_h1"


    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)
        
        if mj_data.time <= 0.01:
            led_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "status_led")
            mj_model.geom_rgba[led_id] = [0, 1.0, 0.0, 1.0]

        # store imu
        # t_store.append(mj_data.time)
        # gyro_store.append(mj_data.sensor("imu_gyro").data.copy())
        # acc_store.append(mj_data.sensor("imu_acc").data.copy())

        # tof_store.append(mj_data.sensor("tof").data.copy())
        
        # control
        match fall:
            case "sit":
                left_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch")
                mj_data.ctrl[left_hip_pitch_id] = -9.0
                right_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch")
                mj_data.ctrl[right_hip_pitch_id] = -9.0
                waist_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch")
                mj_data.ctrl[waist_pitch_id] = 15
            case "sit_h1":
                left_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_joint")
                mj_data.ctrl[left_hip_pitch_id] = -9.0
                right_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch_joint")
                mj_data.ctrl[right_hip_pitch_id] = -9.0
                left_knee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
                mj_data.ctrl[left_knee_id] = 9.0
                right_knee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_joint")
                mj_data.ctrl[right_knee_id] = 9.0
                force_magnitude = 1500  # 100N，约10kg的推力
                direction = np.array([0.34, 0.87, 0])  # 方向
                body_name = "torso_link"
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)      
                if 0 <= mj_data.time <= 0.2:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
            case "45_left":
                force_magnitude = 100  # 100N，约10kg的推力
                direction = np.array([0, 1, 0])  # -Y方向（根据你的模型，这是右侧）
                body_name = "torso_link"
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)      
                if 0 <= mj_data.time <= 2:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
                else:
                    # 其他时间清零，否则力会一直作用
                    mj_data.xfrc_applied[body_id, :3] = 0
            case "45_right":
                force_magnitude = 100  # 100N，约10kg的推力
                direction = np.array([0, -1, 0])  # -Y方向（根据你的模型，这是右侧）
                body_name = "torso_link"
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)      
                if 0 <= mj_data.time <= 2:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
                else:
                    # 其他时间清零，否则力会一直作用
                    mj_data.xfrc_applied[body_id, :3] = 0
            case "lateral_right":
                right_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch")
                mj_data.ctrl[right_hip_pitch_id] = -9.0
                force_magnitude = 100  # 100N，约10kg的推力
                direction = np.array([1, -2.5, 0])  # -Y方向（根据你的模型，这是右侧）
                body_name = "torso_link"
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)      
                if 0 <= mj_data.time <= 2:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
                else:
                    # 其他时间清零，否则力会一直作用
                    mj_data.xfrc_applied[body_id, :3] = 0
            case "lateral_left":
                left_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch")
                mj_data.ctrl[left_hip_pitch_id] = -9.0
                force_magnitude = 100  # 100N，约10kg的推力
                direction = np.array([1, 2.5, 0])  # -Y方向（根据你的模型，这是右侧）
                body_name = "torso_link"
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)      
                if 0 <= mj_data.time <= 2:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
                else:
                    # 其他时间清零，否则力会一直作用
                    mj_data.xfrc_applied[body_id, :3] = 0
            case "forward":
                force_magnitude = 400  # 100N，约10kg的推力
                direction = np.array([1, 0, 0])  # -Y方向（根据你的模型，这是右侧）
                body_name = "torso_link"
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if 0 <= mj_data.time <= 0.05:
                    waist_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch")
                    mj_data.ctrl[waist_pitch_id] = 10
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
                    left_knee_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee")
                    mj_data.ctrl[left_knee_pitch_id] = 4.5
                    right_knee_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee")
                    mj_data.ctrl[right_knee_pitch_id] = 4.5
                    left_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch")
                    mj_data.ctrl[left_hip_pitch_id] = 3
                    right_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch")
                    mj_data.ctrl[right_hip_pitch_id] = 3
                elif 0.05 < mj_data.time <= 0.3:
                    # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
                    mj_data.xfrc_applied[body_id, :3] = force_magnitude * direction
                    
                else:
                    # 其他时间清零，否则力会一直作用
                    mj_data.xfrc_applied[body_id, :3] = 0
                    mj_data.ctrl[:] = 0
            case _:
                print("skippepd")  

        # led decision
        temp_acc = mj_data.sensor("imu_acc").data.copy()
        # print(temp_acc)
        norm = np.sqrt(temp_acc[0]**2 + temp_acc[1]**2 + temp_acc[2]**2)
        led_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "status_led")
        if norm > 80 or mj_data.sensor("tof").data.copy() < 0.62:
            # 红色 RGBA
            mj_model.geom_rgba[led_id] = [1.0, 0.0, 0.0, 1.0]

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    # plotting(t_store,gyro_store,acc_store,tof_store)
    # export_csv(t_store,gyro_store,acc_store,tof_store)
    # print(tof_store)
    


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
