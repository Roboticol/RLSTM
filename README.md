<img width="1919" height="1012" alt="Screenshot_20251230_122901" src="https://github.com/user-attachments/assets/4e7b792a-6eee-4025-b94a-fafebfb03ac5" />

# RLSTM
A C passion project aiming to recreate how Long Short-Term Memory Networks work.
As of now the forward pass functionality is fully complete.

## How to build and run
First, ensure that the [Gnu Scientific Library](https://mirrors.ustc.edu.cn/gnu/gsl/) is installed on your computer. You can either install it from this link or depending on your operating system:

### Linux:
#### Ubuntu/Debian:
```sudo apt install libgsl-dev```
#### Arch:
```sudo pacman -S gsl```
#### Fedora:
```sudo dnf install gsl-devel```

### MacOS:
```brew install gsl```

### Windows:
Install MSYS2 first from [here](https://www.msys2.org). Then,
```pacman -S mingw-w64-x86_64-gsl```

Now to run the program, all you have to do is go to the root folder and run this command:

```make && ./build/main```

Internal structure of the libraries i've written:
<img width="1640" height="1390" alt="4" src="https://github.com/user-attachments/assets/e0b8016d-0876-41e1-819d-f41502341a37" />
<img width="1806" height="1032" alt="3" src="https://github.com/user-attachments/assets/f269d8e9-fabd-4a7a-a766-cb9bffbc05f6" />
<img width="1536" height="1000" alt="5" src="https://github.com/user-attachments/assets/6fe9dbb4-3577-4bb7-9384-2a381900657d" />
