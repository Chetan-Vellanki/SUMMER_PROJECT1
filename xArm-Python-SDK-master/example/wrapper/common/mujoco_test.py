import mujoco
import mujoco.viewer
import numpy as np
import time


dt : float = 0.002

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("/Users/sasivellanki/Desktop/xArm-Python-SDK-master/example/wrapper/common/mjctrl/ufactory_xarm7/scene_strike_act.xml")
    data = mujoco.MjData(model)

    site_name = "link_tcp"
    site_id = model.site(site_name).id

    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos
    q01 = q0[:7]
    jac = np.zeros((6, model.nv))
    time_start = time.time()
    while (True) : 
        step_start = time.time()
        data.ctrl[actuator_ids] = q0[dof_ids]
        mujoco.mj_step(model, data)
        # viewer.sync()
        

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        if(time.time() - time_start > 1) : 
            break

    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    jac1 = jac[:,:7]
    print("Jacobian : ", jac1)

            

if __name__ == "__main__":
    main()