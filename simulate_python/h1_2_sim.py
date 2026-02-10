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

def export_csv(csv_name, t_store, gyro_store, acc_store, tof_store):

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
        csv_name,
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

def random():
    # randomness
    direction = np.array([
        np.random.uniform(-1, 1),    
        np.random.uniform(-1, 1),    
        0  
    ])
    direction = np.round(direction, 3)
    force_mag = np.random.randint(700,1500)
    
    return direction, force_mag


def SimulationThread():
    global mj_data, mj_model

    # data
    t_store   = []
    gyro_store = []   # 3 轴角速度
    acc_store  = []   # 3 轴加速度
    tof_store = []
    duration = []
    
    #randomness
    direction, force_mag = random()

    episode = 0
    

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

       


        # store imu
        # t_store.append(mj_data.time)
        # gyro_store.append(mj_data.sensor("imu_gyro").data.copy())
        # acc_store.append(mj_data.sensor("imu_acc").data.copy())

        # tof_store.append(mj_data.sensor("tof").data.copy())
        
        # control
        #left_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_joint")
        #mj_data.ctrl[left_hip_pitch_id] = -9.0
        #right_hip_pitch_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch_joint")
        #mj_data.ctrl[right_hip_pitch_id] = -9.0
        #left_knee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint")
        #mj_data.ctrl[left_knee_id] = 9.0
        #right_knee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_joint")
        #mj_data.ctrl[right_knee_id] = 9.0
        
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)      
        if 0 <= mj_data.time <= 0.15:
            # xfrc_applied[body_id, 0:3] 是力，[3:6] 是力矩
            mj_data.xfrc_applied[body_id, :3] = force_mag * direction

        pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        # print(mj_data.xpos[pelvis_id][2])
        if mj_data.xpos[pelvis_id][2] <= 0.25 or mj_data.time >= 10:
            direction, force_mag = random()
            duration.append(mj_data.time)
            mujoco.mj_resetData(mj_model, mj_data)  
            episode = episode + 1   
            
        
        if episode >= 10:
            print(duration)
            viewer.close()     


        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    


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
