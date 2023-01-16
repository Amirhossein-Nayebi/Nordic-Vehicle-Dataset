# How to run FFD
FFD is a Linux executable. To run it on Windows, one way is to use the Windows Subsystem for Linux (WSL) and install Ubuntu 18.04 on it. Please note that you have to install Ubuntu-18.04 not the newer versions. Here are the instructions:

Open a terminal and run:

    wsl --update 
    wsl --install -d ubuntu-18.04

After installing Ubuntu 18.04 and creating a user, you need to install the C++ version of OpenCV 3.2 library:

    sudo apt-get update
    sudo apt-get install libopencv-dev

Now, you should be able to run the 'FFD' linux executable on Ubuntu-18.04 distribution of WSL. There is a ```test.py``` python script that automate the process of executing the file with the required parameters and retrieving the results.
