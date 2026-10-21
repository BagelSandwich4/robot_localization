# Robot localization
Written by Ophelia Lonzo, Thanadetch (Detch) Mateedunsatits

For a more detailed writeup, please read the Writeup.pdf

[Technical report and writeup](Writeup.pdf) 

Below describes how to run the code, which is also found at the end of the writeup

# How To Run
To run the particle, first get the code and build it:

```
cd ~/ros2_ws/src
git clone https://github.com/BagelSandwich4/robot_localization.git robot_localization
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Use three terminals:

Terminal 1: the particle filter and the map server

```
ros2 launch robot_localization test_pf.py map_yaml:=/home/<your-username>/ros2_ws/src/robot_localization/maps/mac_1st_floor_final.yaml
```


Terminal 2: rviz
```
rviz2 -d ~/ros2_ws/src/robot_localization/rviz/turtlebot_bag_files.rviz
```

Terminal 3: the bag file
```
ros2 bag play ~/ros2_ws/src/robot_localization/bags/macfirst_floor_take_1 --clock --start-paused
```

1. In Terminal 3, press the spacebar to start the bag. Let it run for 2 to 3 seconds.
2. Press the spacebar again to pause it.
3. Press the spacebar in Terminal 3 to resume the bag.

*It is also possible to play other bag files by passing it into terminal 3 and replacing the name of the file being played from macfirst_floor_take_1 to the bag file name (see bag file folders).

