# How to Install & Run AutoClip Studio 🚀

Hey! Follow these simple, step-by-step instructions to get AutoClip Studio running on your Windows machine. 

## Step 1: Install Python
You need Python to run the code.
1. Go to the [Official Python Website](https://www.python.org/downloads/).
2. Download the latest installer for Windows.
3. **CRITICAL STEP:** When you open the installer, make sure you check the box at the bottom that says **"Add python.exe to PATH"** before clicking "Install Now".
4. Once installed, open your Command Prompt (search `cmd` in the Windows start menu) and type `python --version` to verify it installed correctly.

## Step 2: Install FFmpeg
FFmpeg is a tool used to process video and audio under the hood.
1. Open your Command Prompt (search for `cmd` in the Start menu).
2. Type the following command and hit Enter:
   ```bash
   winget install Gyan.FFmpeg
   ```
3. Let it finish downloading and installing.

## Step 3: Download AutoClip Studio
1. Download this entire folder and extract it to your Desktop or Documents.
2. Open the folder so you can see all the files like `Start_AutoClip.bat` and `requirements.txt`.

## Step 4: Install Dependencies
1. Inside the AutoClip Studio folder, click on the address bar at the very top of the File Explorer (where it shows the folder path).
2. Delete the text, type `cmd`, and hit **Enter**. This opens a Command Prompt directly in that folder.
3. Type this command and hit Enter to install all the AI models and required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(This might take a few minutes as it downloads PyTorch and Whisper packages!)*

## Step 5: Start the App!
1. Once the installation is finished, you can simply double-click the **`Start_AutoClip.bat`** file inside the folder.
2. A black terminal window will pop up to start the server, and then it will automatically open your web browser to `http://127.0.0.1:8000`.

### Troubleshooting
- **"Python not found" error**: You forgot to check "Add Python to PATH" in Step 1. Uninstall Python, reinstall, and make sure to check that box!
- **Download fails or stays stuck**: Make sure you have a stable internet connection. 
- **The very first run**: The very first time you generate a video, it might seem stuck at the "Transcribe" step for a few minutes. It is downloading the large AI model in the background. Subsequent runs will be much faster.

Enjoy making viral clips! 🎉
