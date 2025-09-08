
Glass:
  Translate: -0.25, 0.4,0.8
  orient: -90,50,-180
Robot:
  Translate: -0.15,-0.05,0.76


NUCLEUS_ASSET_ROOT_DIR = ("/home/app/unitree/isaacsim_assets45/Assets/Isaac/4.5")
NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"

e.g.
usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
/home/app/unitree/isaacsim_assets45/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd
{project_root}/assets/objects/PackingTable/PackingTable.usd
/home/app/unitree/unitree_sim_isaaclab/assets/objects/PackingTable/PackingTable.usd

1) start isaac sim 4.5
conda activate unitree_sim_env
cd /home/app/unitree/unitree_sim_isaaclab

e.g. python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex3-Joint   --enable_dex3_dds --robot_type g129

python sim_main.py --device cpu  --enable_cameras  --task  Nsb-Paint-Cylinder-G129-Dex1-Joint   --enable_dex1_dds --robot_type g129
python sim_main.py --device cuda  --enable_cameras  --task  Nsb-Paint-Cylinder-G129-Dex1-Joint   --enable_dex1_dds --robot_type g129

python sim_main.py --device cpu  --enable_cameras  --task  Nsb-Paint-Cylinder-G129-Dex3-Joint   --enable_dex3_dds --robot_type g129
python sim_main.py --device cuda  --enable_cameras  --task  Nsb-Paint-Cylinder-G129-Dex3-Joint   --enable_dex3_dds --robot_type g129

python sim_main.py --device cpu  --enable_cameras  --task  Nsb-Paint-Cylinder-G129-Inspire-Joint   --enable_inspire_dds --robot_type g129
python sim_main.py --device cuda  --enable_cameras  --task  Nsb-Paint-Cylinder-G129-Inspire-Joint   --enable_inspire_dds --robot_type g129


2) start xr_teleop
conda activate tv
cd /home/app/unitree/xr_teleoperate/teleop
python teleop_hand_and_arm.py --ee=inspire1 --sim --record
