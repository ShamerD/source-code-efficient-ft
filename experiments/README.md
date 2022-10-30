### Reproducing Figures and Tables

Follow the table to reproduce the needed element.

| Element   | Experiment IDs                     |
|-----------|------------------------------------|
| `Fig. 1a` | `[5-6, 13, 17, 21, 25]`            |
| `Fig. 1b` | `[11, 48-51]`                      |
| `Fig. 1c` | `[7-8, 14, 18, 22, 26, 29-31, 56]` |
| `Fig. 1d` | `[9, 15, 19, 23, 27, 67-70]`       |
| `Fig. 1e` | `[71-72, 79, 83, 87]`              |
| `Fig. 1f` | `[75, 78, 82, 86, 90]`             |
| `Fig. 1g` | `[73, 76, 80, 84, 88, 111-114]`    |
| `Fig. 1h` | `[74, 77, 81, 85, 89, 115-118]`    |
| `Fig. 2a` | `[5-6]`                            |
| `Fig. 2b` | `[7-8]`                            |
| `Table 2` | `[32-33, 35, 52, 119-124]`         |
| `Fig. 3a` | `[57-61]`                          |
| `Fig. 3b` | `[101-105]`                        |
| `Fig. 3c` | `[62-66]`                          |
| `Fig. 3d` | `[91-95]`                          |
| `Fig. 3e` | `[106-110]`                        |
| `Fig. 3f` | `[96-100]`                         |
| `Table 3` | `[44, 125-133]`                    |
| `Table 4` | `[36, 134-142]`                    |
| `Table 5` | `[40, 143-151]`                    |
| `Table 6` | `[152-161]`                        |
| `Table 7` | `[162-171]`                        |
| `Table 8` | `[172-181]`                        |

### Running without SLURM
1. These tasks were used in SLURM task manager and contain specific SLURM syntax.
2. Called script is placed after `#Executable`.
3. `TRANSFORMERS_OFFLINE=1` is needed if there is no internet connection (model should be downloaded manually), and can be safely deleted otherwise.
4. The only SLURM-specific thing after `#Executable` is `$idx` variable (environmental) which were used to parallelize runs. It should be replaced with needed parameter.
