# NeuroEdge Compiler 🧠⚡
> Adaptive Neural Network Compiler with Hardware-Aware Optimization

## 10-Week Development Log

| Day | Feature | Status | Key Result |
|-----|---------|--------|------------|
| Day 1 | INT4 Quantization | ✅ Done | 8x compression vs FP32 |
| Day 2 | Sparsity Analyzer | ✅ Done | 100% sparsity at t=0.50 |
| Day 3 | Layer-wise Stats | ✅ Done | 232K params analyzed |
| Day 4 | Weight Histogram + CDF | ✅ Done | Per-layer distribution plots |
| Day 5 | Mixed Precision INT4+INT8 | ✅ Done | 7.77x overall compression |
| Day 6 | Pruning Threshold Optimizer | ✅ Done | Binary search → 70% sparsity |
| Day 7 | Week 1 Summary Plot | 🔄 Next | |
| Day 8–70 | Coming soon... | ⏳ | |

## Features
- INT4 / INT8 / Mixed Precision Quantization
- Sparsity Analysis with threshold sweep
- Layer-wise compression statistics
- Weight distribution visualization
- Pruning threshold optimizer (binary search)

## Results So Far
| Metric | Value |
|--------|-------|
| Max compression | 8x (INT4 vs FP32) |
| Mixed precision | 7.77x with quality control |
| Pruning target | 70% sparsity achieved |
| Layers analyzed | conv1, conv2, fc1, fc2, output |

## Setup
```bash
conda activate neuroedge
python compiler/quantize_int4.py
python compiler/sparsity_analyzer.py
python compiler/layer_stats.py
python compiler/histogram_plotter.py
python compiler/mixed_precision.py
python compiler/pruning_optimizer.py
