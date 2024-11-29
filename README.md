# Vein Cracker

Given details about a Minecraft vein (such as its coordinates, type, and attributes), this repository is designed to return a list of internal java.util.Random states that could <ins>potentially</ins> generate that exact vein. With multiple veins, you can then use those internal states to find which lowest 48 bits of worldseeds could <ins>potentially</ins> generate all of them simultaneously.

## Prerequisites and Limitations
This program formally supports
- Java Edition.
- Versions Beta 1.6&mdash;1.12.2.
- Dirt, gravel, coal, iron, gold, redstone, or diamond veins.

This program uses CUDA, which requires your device to have an NVIDIA CUDA-capable GPU installed. NVIDIA's CUDA also [does not support MacOS versions OS X 10.14 or beyond](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts). If either of those requirements disqualify your computer, you can instead run the program on a virtual GPU for free (at the time of writing this, and subject to certain runtime limits) through [Google Colab](https://colab.research.google.com).

If using Windows, you will also need some form of C++ compiler installed; however, there are a myriad of environments that provide one ([Microsoft Visual C++](https://learn.microsoft.com/en-us/cpp/build/reference/compiler-options), though that in turn requires [Visual Studio](https://visualstudio.microsoft.com); [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl); [Minimial System 2](https://www.msys2.org); and others).

## Installation, Setup, and Usage
1. Download the repository, either as a ZIP file from GitHub (which you then extract) or by cloning it through Git.
2. Open [the Settings file](./Settings%20(MODIFY%20THIS).cuh) in your favorite code editor, and replace the examples of input data with your own, and the settings with your own. (For enumerations like `Version`, the list of supported values can be found in [Allowed Values for Settings.cuh](./Allowed%20Values%20for%20Settings.cuh).)

(Note: If you are looking for the lowest 48 bits of worldseeds, at bare minimum two veins will be necessary: this will currently require running the program multiple times, once for each vein. If multi-day runtimes cannot be avoided, one setting the program comes with is the ability to divide your runs into "partial runs", so that a run can be restarted midway-through at a later time.)

3. Go back and double-check your input data. There is an 80% chance you inputted something incorrectly the first time, and any mistakes will prevent the program from deriving the correct internal states, and by extension, the correct lowest 48 bits of worldseeds.
4. Once you're *completely certain* your input data is correct&mdash;if you wish to run the program on Google Colab:
    1. Visit [the website](https://colab.research.google.com), sign in with a Google account, and create a new notebook.
    2. Open the Files sidebar to the left and upload the program's files, making sure to keep the files' structure the way it originally was (the underlying code files are inside a folder named src, etc.). Don't forget to upload the modified Settings file instead of the original.
    3. Under the Runtime tab, select "Change runtime type" and select T4 GPU as the hardware accelerator.
5. Whether on Google Colab or your own computer, open a terminal and verify [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html), the CUDA compiler, is installed:
```bash
(Linux/Windows/MacOS)  nvcc --version
(Google Colab)        !nvcc --version
```
If the output is an error and not the compiler's information, you will need to install the CUDA Toolkit which contains `nvcc`. (The following are installation guides for [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux), [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows), and [MacOS X 10.13 or earlier](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-mac-os-x/).)

6. Navigate in the terminal to the folder where the program's files are contained:
```bash
(Linux/Windows/MacOS)  cd "[Path to the folder]"
(Google Colab)        !cd "[Path to the folder]"
```
Then use `nvcc` to compile the program:
```bash
(Linux)         nvcc "main.cu" -o "main" -O3
(Windows)       nvcc "main.cu" -o "main.exe" -O3
(MacOS)         nvcc "main.cu" -o "main.app" -O3
(Google Colab) !nvcc "main.cu" -o "main" -O3
```
Depending on your input data, the compilation may take almost a full minute or even longer.<br />
The compiler may print warnings akin to `Stack size for entry function '_Z11biomeFilterv' cannot be statically determined`: this is normal. (All this means is that the compiler couldn't determine the exact number of iterations certain recursive functions will undergo.)

7. Run the compiled program:
```bash
(Linux)         .\main
(Windows)       .\main.exe
(MacOS)         open -a main.app
(Google Colab) !.\main
```
As mentioned in step 2, the program's runtime can vary wildly based on one's input data and its comprehensiveness. Nevertheless, if all goes well, a list should gradually be printed to the screen containing the possible internal states.
8. Those internal states, and other information about the veins, can then be plugged into [Combination.cu](<./Combination.cu>) to retrieve which possible lowest 48 bits of worldseeds could generate the vein(s).
<!-- 
8. At some point, this program will also automatically filter structure seeds into potential worldseeds. This hasn't been implemented yet, though, so in the meantime one must perform this filtering manually.
    1. Download and open [Cubiomes Viewer](https://github.com/Cubitect/cubiomes-viewer/releases).
    2. Under the Edit tab in the upper top-left, click "Advanced World Settings" and make sure "Enable experimentally supported versions" is enabled.
    3. Close the World Settings menu and set the "MC" input box in the top-left corner to your world's version (or the supported version closest to it).
    4. Under the Seed Generator heading, Click "Seed list", then use the button across from the "Load 48-bit seed list" option to select whichever file contains this program's outputted structure seeds.
    5. For each population chunk in your input data (these will have been displayed when <ins>this</ins> program first began running):
        - Under the Conditions heading, click "Add".
        - Select "Biomes" for the condition's category and "Overworld at scale" as the condition's type.
        - Select Custom for the location and enter the population chunk's coordinate range.
        - Select "1:1 ..." for the Scale/Generation Layer, then exclude all biomes except the population chunk's biome.
    6. When finished adding all conditions, click "Start search" at the bottom of the window. The program will then start outputting worldseeds that have biomes matching your input data.

WARNING: When checking the outputted worldseeds, some generated trees may not match your input data. (Tree generation depends on the order that chunks are loaded, so if the chunks are loaded in a different order than your input data's source, a different pattern of trees will form.) However, in most cases at least a few trees will match your input data; if *every* tree is different, that is an indication your original input data (or this tool) are likely wrong. -->

## Acknowledgements
I would like to give very large Thank You's to
- [Andrew](https://github.com/Gaider10), for creating [his TreeCracker](https://github.com/Gaider10/TreeCracker) which this code is partially derived from<!-- and a [population chunk reverser](https://github.com/Gaider10/PopulationCrr), and for answering a question about his tool -->.
<!-- - [Cubitect](https://github.com/cubitect), for his [Cubiomes library](https://github.com/Cubitect/cubiomes) that this program (will ultimately) use a port of to filter biomes, and his [Cubiomes Viewer](https://github.com/Cubitect/cubiomes-viewer) GUI tool I recommend as a substitute in the meantime. -->
- [KaptainWutax](https://github.com/KaptainWutax), for creating [his Kaktoos searcher](https://github.com/KaptainWutax/Kaktoos) which this code is also partially derived from.
- [Panda4994](https://github.com/panda4994), for [his algorithm]((https://github.com/Panda4994/panda4994.github.io/blob/48526d35d3d38750102b9f360dff45a4bdbc50bd/seedinfo/js/Random.js#L16)) to determine if a state is derivable from a nextLong call.

If you would like to contribute to this repository or report any bugs, please feel free to open an issue or a pull request.

This repository is offered under [my (NelS') general seedfinding license](./LICENSE). Please read and abide by that text if you have any wishes of referencing, distributing, selling, etc. this repository or its code.[^1]

<!-- [^1]: If one converts a worldseed into a 64-bit binary integer, a structure seed corresponds to the worldseed's last 48 bits. Therefore each structure seed has 2<sup>16</sup> = 65536 worldseeds associated with it. ...eventually, biome filtering will be used to directly return worldseeds instead of structure seeds, but this has not been finished yet. -->
[^1]: While the license discusses this, I want to emphasize one aspect of it here: this repository relies upon numerous others' repositories (Gaider10's TreeCracker, KaptainWutax's Kaktoos code, etc.), and thus my license solely applies to the changes I and any voluntary contributors made within this repository, not to their repositories or any code in this repository that is untouched from their repositories.