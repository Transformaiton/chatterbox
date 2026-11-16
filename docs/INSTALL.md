# Install Chatterbox TTS

This is the install guide for [Transformaiton.com](https://transformaiton.com/)'s public copy of Chatterbox. It covers Windows, Mac, and Linux. Generation examples live in [USAGE.md](USAGE.md). The upstream technical README is preserved in [ORIGINAL.md](ORIGINAL.md).

The model code is unchanged. These steps install that code and the libraries it imports.

## Before you install

1. **Python 3.10 or newer.** Upstream developed and tested Python 3.11 on Debian 11. Use 3.11 if you can choose. Python 3.14 needs a newer PyTorch than 2.6; the project file already selects that version when it sees 3.14.
2. **Git.** Required even for a PyPI install, because `resemble-perth` is fetched from GitHub.
3. **Disk.** Leave about 10 GB free. PyTorch and the other libraries take a few gigabytes. The English checkpoints `t3_cfg.safetensors` and `s3gen.safetensors` add about 3.2 GB. Nano's download is about 3 GB because the loader fetches every `.safetensors` file in `ResembleAI/chatterbox-nano`, including files it does not load into memory. Weights are cached in your user folder under `.cache/huggingface`.
4. **Memory.** Nano's loaded files are a bit under 2 GB. The English model's loaded files are about 3.2 GB. A computer with 16 GB of RAM is a sensible starting point. An 8 GB computer may be killed by the operating system once generation starts.
5. **A GPU is optional.** See the [CPU or GPU](../README.md#cpu-or-gpu) section of the README. Nano is the model upstream aimed at CPU machines.

Confirm the tools:

```bash
python3 --version
git --version
```

On Windows, `py -3.11 --version` is the command to check after you install Python.

## Windows

1. Download **Python 3.11** from [python.org/downloads](https://www.python.org/downloads/). On the first installer screen, turn on **Add python.exe to PATH**.
2. Download **Git** from [git-scm.com/download/win](https://git-scm.com/download/win) and accept the default options. This also provides Git Bash.
3. Open **PowerShell**.
4. If activating a virtual environment is blocked, run this once in that window. It lasts only for the current window:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

5. Clone this repository and install it:

   ```powershell
   git clone https://github.com/Transformaiton/chatterbox.git
   cd chatterbox
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -U pip
   pip install -e .
   ```

6. Check that the library imports:

   ```powershell
   python -c "import chatterbox, torch; print(chatterbox.__version__, torch.__version__)"
   ```

`pip install -e .` compiles a few small packages if a wheel is missing. If that step asks for **Microsoft C++ Build Tools**, install the "Desktop development with C++" workload from the Visual Studio Build Tools installer, then run `pip install -e .` again in the same activated environment.

The first speech command, not the install, downloads the model. Continue with [USAGE.md](USAGE.md).

## Mac

1. Install Apple's command-line tools. In Terminal:

   ```bash
   xcode-select --install
   ```

2. Install Python 3.11 if `python3 --version` is older than 3.10. The installer at [python.org](https://www.python.org/downloads/macos/) works. Homebrew is the other common path:

   ```bash
   brew install python@3.11 git
   ```

3. Clone and install:

   ```bash
   git clone https://github.com/Transformaiton/chatterbox.git
   cd chatterbox
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   pip install -e .
   python -c "import chatterbox, torch; print(chatterbox.__version__, torch.__version__)"
   ```

   If the `python3.11` command does not exist, use the `python3` that reports 3.11 or newer.

4. Apple Silicon can use the GPU via MPS. The cross-platform quick start in the README selects `mps` when PyTorch reports it. If loading weights raises a device error, use the included Mac example. Open `example_for_mac.py` and set `AUDIO_PROMPT_PATH` to a WAV you have permission to use, then run:

   ```bash
   python example_for_mac.py
   ```

   That file patches `torch.load` so checkpoints map onto the Mac. Intel Macs have no MPS device, so the same scripts use the CPU.

## Linux

These commands match Debian and Ubuntu, including the Debian 11 setup upstream used. On Fedora, the package names differ (`python3`, `python3-devel`, `git`, `ffmpeg`, and the compiler group).

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev git ffmpeg build-essential
```

Install Python 3.11 from your distribution if `python3 --version` is below 3.10. Then:

```bash
git clone https://github.com/Transformaiton/chatterbox.git
cd chatterbox
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
python -c "import chatterbox, torch; print(chatterbox.__version__, torch.__version__)"
```

A Conda environment also works. Upstream's note was:

```bash
conda create -yn chatterbox python=3.11
conda activate chatterbox
cd chatterbox
pip install -e .
```

Create the conda environment from the repository directory after the clone, or `cd` into it before `pip install -e .`.

An NVIDIA GPU needs a current NVIDIA driver so PyTorch can see it. Check with `nvidia-smi`. You do not need the CUDA toolkit just to run. If `torch.cuda.is_available()` is false, the quick start uses the CPU. CPU-only PyTorch wheels are smaller if you install them yourself before `pip install -e .`:

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Use that pin on Python 3.13 and earlier. On Python 3.14 the project asks for PyTorch 2.9 or newer, so skip the pin and let `pip install -e .` choose.

## Install from PyPI

```bash
pip install chatterbox-tts
```

That command installs the published package named `chatterbox-tts`. Installing this clone with `pip install -e .` uses the files you just downloaded from [github.com/Transformaiton/chatterbox](https://github.com/Transformaiton/chatterbox). Use the clone when you want the examples and browser apps that sit next to this guide.

## Confirm the install

With the virtual environment active:

```bash
python -c "import chatterbox, perth, torch; print('chatterbox', chatterbox.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

A successful install prints a version line and `cuda False` on a machine without an NVIDIA GPU. Speech is the next step, and it is the step that downloads weights. Follow [USAGE.md](USAGE.md).

## Install troubleshooting

**`python3 -m venv` says `ensurepip` is not available.** Install the venv package (`sudo apt install python3-venv` or `python3.12-venv` on Ubuntu) and create the environment again.

**`git` is not recognized.** Install Git, then open a new terminal so PATH updates. On Windows, Git Bash is a reliable place to rerun the clone.

**Permission denied on `pip install`.** You are outside the virtual environment, or the environment was created with another user. Activate `.venv` and install there. Do not use `sudo pip`.

**`resemble-perth` fails with an authentication or `git+https` error.** Confirm `git ls-remote https://github.com/resemble-ai/Perth.git` works on that network. The dependency line in `pyproject.toml` points at that repository's `master` branch.

**A wheel fails to build (`pkuseg`, `conformer`, or similar).** Install the compiler packages in the Linux section, or the Microsoft C++ Build Tools on Windows, then rerun `pip install -e .`.

**`No module named chatterbox` after a successful pip.** The shell that runs the script is not the shell where `.venv` is active. Activate it again. The prompt usually starts with `(.venv)`.

**Disk fills up during the first speech run.** Weights land in `~/.cache/huggingface` (on Windows, `C:\Users\<you>\.cache\huggingface`). Remove that cache only when you are willing to download again. The repository folder itself stays small.

**Antivirus locks files on Windows.** Allow the Python in `.venv` and retry `pip install -e .`.
