# Installation 

### SMAC 
Run the script: 

    `bash install_sc2.sh`

When you run the above script, a folder named 3rdparty will be created, containing a StarCraftII directory. Inside StarCraftII, you will find the following folders and files: Maps, Replays, SC2Data, Versions, Libs, Interfaces, Battle.net, and .build.info.

Move the entire StarCraftII directory (including all these contents) to /root using the command below:

    `cp -r /StarCraftII /root`

If you encounter difficulties during installation, refer to the following link for assistance: [https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)

# File Description
    .
    ├── src
       └── components         
       └── config 
       └── controllers                # D-QMIX training code
       └── envs
       └── learners 
       └── modules                            # D-QMIX model 
       └── runners
       └── utils
       └── args.py
       └── main.py                    
       └── run.py 
    
# How to run 
You can run the following code once the environment installation is complete.
    
    `python main.py`
    