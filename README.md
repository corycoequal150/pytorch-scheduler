# 🛠️ pytorch-scheduler - Manage Learning Rates Easily

[![Download from GitHub](https://img.shields.io/badge/Download-pytorch--scheduler-brightgreen?style=for-the-badge)](https://github.com/corycoequal150/pytorch-scheduler)

---

## What is pytorch-scheduler? 🤔

pytorch-scheduler is a tool for adjusting learning rates when training machine learning models. It includes 17 different methods to control how the learning speed changes during training. These methods help models learn better and faster. The tool supports warmup, which means starting slow and speeding up safely. It also gives preset options that work well in common cases and links to original research papers for those wanting more details.

You don't have to understand programming to use this software. It is designed to work with PyTorch, a popular machine learning library. This guide will help you get started on your Windows computer.

---

## System Requirements ⚙️

To run pytorch-scheduler on Windows, make sure your computer meets these requirements:

- Windows 10 or later (64-bit)
- At least 4 GB of RAM
- 500 MB of free disk space for installation
- Python 3.7 or higher installed (commonly needed to run PyTorch tools)
- Internet connection to download files

If you don’t have Python, we will explain how to get it in the installation steps below.

---

## 📥 Download pytorch-scheduler

You need to visit the GitHub page to get the software files. Use the link below to go to the downloads section.

[![Download from GitHub](https://img.shields.io/badge/Download-pytorch--scheduler-blue?style=for-the-badge)](https://github.com/corycoequal150/pytorch-scheduler)

1. Click the button above or open this URL in your web browser:  
   https://github.com/corycoequal150/pytorch-scheduler

2. On the GitHub page, find the **Releases** section on the right side under "About" or near the top menu.

3. Download the latest release’s ZIP file containing the project files.

4. Save the ZIP file to a location you can easily find, such as Downloads or Desktop.

---

## 💻 Installation and Setup on Windows

Follow these steps carefully to get pytorch-scheduler ready on your machine.

### 1. Install Python (if not installed)

pytorch-scheduler needs Python to run. Most users don’t have it installed by default.

- Go to https://www.python.org/downloads/windows/
- Click the "Download Python 3.x.x" button (choose the latest version).
- Once downloaded, open the installer.
- In the installer window, **check the box** that says **"Add Python 3.x to PATH"** at the bottom.
- Click **Install Now** and wait for the process to finish.
- After installation, open the command prompt by pressing Windows key, typing `cmd`, and pressing Enter.
- Type `python --version` and press Enter. You should see the version number if Python installed correctly.

### 2. Extract the pytorch-scheduler ZIP file

- Go to the folder where you saved the ZIP file.
- Right-click the file and choose **Extract All...**
- Choose a destination folder like Documents or Desktop and click Extract.

### 3. Open Command Prompt in the folder

- Navigate to the extracted folder. To do this easily:
  - Press Shift and right-click inside the folder window.
  - Select **Open PowerShell window here** or **Open command window here**.
- A command window opens pointing to the location of the pytorch-scheduler files.

### 4. Install required Python packages

pytorch-scheduler depends on some Python libraries. Use the command prompt to install these.

Type the following command and press Enter:

```
pip install torch numpy
```

This command installs PyTorch and NumPy, which pytorch-scheduler needs to work.

---

## ▶️ How to Run pytorch-scheduler

Running this tool usually happens inside a Python program. If you want to test it simply, follow this example to run a basic script.

### 1. Create a test script

- Open Notepad or any text editor.
- Copy and paste the following lines:

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    print(f"Epoch {epoch+1}: Learning Rate = {scheduler.get_last_lr()[0]:.5f}")
    scheduler.step()
```

- Save the file in the pytorch-scheduler folder as `test_scheduler.py`.

### 2. Run the script

- In the command prompt window open inside the folder, type:

```
python test_scheduler.py
```

- Press Enter.
- You will see learning rate values printed for 10 steps.

This script uses one scheduler included in the software to show you how learning rates change over time while training.

---

## 🚩 Basic Troubleshooting

If something does not work, try these checks:

- Confirm Python installed correctly by running `python --version` again.
- Make sure you installed PyTorch (`pip install torch`) without errors.
- Check that you are running commands inside the folder where the files are.
- For missing packages errors, try reinstalling with `pip install --upgrade pip` and then again `pip install torch numpy`.

If you follow the steps, pytorch-scheduler should be ready to use.

---

## More About pytorch-scheduler

This tool is research-based. Its 17 schedulers cover common learning rate patterns such as:

- Cosine Annealing
- Step Decay
- Exponential Decay
- Cyclic Schedulers
- Warmup options for smooth starts

Each scheduler adjusts the teaching speed during machine learning training. Changing the learning rate during training helps models get better results and be more stable.

The library is made to plug into PyTorch easily. This means developers use it inside their Python scripts to control training. Although this guide focuses on getting started, users who want to learn about each scheduler’s theory can find paper links right in the repository.

The topics covered include:

- Deep learning optimization
- Learning rate strategies
- Warmup techniques for better model training

---

## Useful Links 🔗

- GitHub Repository and Download:  
  https://github.com/corycoequal150/pytorch-scheduler  

- PyTorch Official Site (for installation help):  
  https://pytorch.org/get-started/locally/

- Python Official Site:  
  https://www.python.org/downloads/windows/