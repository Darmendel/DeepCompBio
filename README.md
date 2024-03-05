# Predicting RNAcompete Binding from RNA Bind-n-Seq Data Project

This project aims to predict RNAcompete binding outcomes based on RNA Bind-n-Seq data. Follow the guidelines below to effectively run the project:

## Requirements:

- Python version 3.7-3.9.
- Required packages are listed in the `requirements.txt` file.

## How to Run:

To run the project, follow these steps:

1. Open your terminal.
2. Navigate to the project directory.

Run the following command:

```bash
python3 main.py <ofile> <RNCMPT> <input> <RBNS1> <RBNS2> ... <RBNSk>
```

Replace the placeholders with actual file paths:

- `<ofile>`: Specify the path to the output file where results will be saved.
- `<RNCMPT>`: Provide the path to the `RNAcompete_sequence.txt` file.
- `<input>`: Set the path to the `RBPi_input.seq` file.
- `<RBNS1>`, `<RBNS2>`, ..., `<RBNSk>`: Replace these with paths to the RBNS files corresponding to different concentrations (j1nM, j2nM, ..., jknM).

Ensure you have the necessary files in the specified locations before running the command.

Example command:

```bash
python3 main.py results/output.txt data/RNAcompete_sequence.txt data/RBPi_input.seq data/RBPi_j1nM.seq data/RBPi_j2nM.seq ... data/RBPi_jknM.seq
```

This command will execute the main script and perform RNAcompete binding prediction based on the provided data.

Feel free to adjust the command according to your file paths and preferences. Good luck with your RNA binding prediction project!
