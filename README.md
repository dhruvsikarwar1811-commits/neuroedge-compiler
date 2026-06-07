# NeuroEdge — Adaptive Neural Hardware Compiler

**B.Tech Final Year Project | ECE | Manipal University Jaipur**
**Author: Yuvraj Sikarwar | 2022-2026**

---

## What is NeuroEdge?

NeuroEdge is an end-to-end adaptive compiler that takes a trained neural network and automatically generates optimized hardware-ready output for edge devices — without manual RTL coding.

## Pipeline## Results

| Metric | Value |
|--------|-------|
| Model Accuracy | 98.22% |
| Size Reduction | 11.2x smaller |
| Weight Pruning | 30% weights removed |
| Target | FPGA / Edge Device |

## Project Structure## How to Run

```bash
conda activate neuroedge
cat > README.md << 'ENDOFFILE'
# NeuroEdge — Adaptive Neural Hardware Compiler

**B.Tech Final Year Project | ECE | Manipal University Jaipur**
**Author: Yuvraj Sikarwar | 2022-2026**

## What is NeuroEdge?

NeuroEdge is an end-to-end adaptive compiler that takes a trained neural network and automatically generates optimized hardware-ready output for edge devices — without manual RTL coding.

## Pipeline

Neural Network → Analyze → Prune → Quantize → RTL Generation

## Results

| Metric | Value |
|--------|-------|
| Model Accuracy | 98.22% |
| Size Reduction | 11.2x smaller |
| Weight Pruning | 30% weights removed |
| Target | FPGA / Edge Device |

## Project Structure

neuroedge-compiler/
├── models/          # CNN training and conversion
├── compiler/        # Optimization engine
│   ├── model_analyzer.py
│   ├── quantizer.py
│   └── pruner.py
├── rtl/             # Verilog RTL modules
│   └── mac_unit.v
├── testbench/       # RTL simulation
└── main.py          # Single entry point

## How to Run

conda activate neuroedge
python3 main.py

## Technologies

- Python 3.10, TensorFlow 2.13
- Verilog HDL, RTL Design
- INT8 Quantization, Weight Pruning
- FPGA Target: Xilinx Basys3

## Key Features

- Auto model profiling and layer analysis
- INT8 quantization (11x size reduction)
- Magnitude-based weight pruning (30% sparsity)
- Auto-generated synthesizable Verilog RTL
- End-to-end pipeline with single command
