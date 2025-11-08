# pytest-gpu-monitor

A pytest plugin to automatically monitor and report GPU memory usage during test execution.

## 🚀 Features

- ✅ **Zero code modification** - Works automatically with `--gpu-monitor` flag
- 📊 **Interactive HTML reports** - Sortable, searchable, and filterable
- 🔍 **Detailed metrics** - Peak memory, duration, memory increase, and more
- 📈 **Multiple formats** - HTML, JSON, Markdown, and CSV reports
- 🎯 **Easy integration** - Simple command-line interface
- 🔥 **Top consumers** - Automatically identifies memory-intensive tests

## 📦 Installation

### Option 1: Install from PyPI (when published)

```bash
pip install pytest-gpu-monitor
```

### Option 2: Install from source

```bash
# Clone the repository
git clone https://github.com/FateScript/pytest-gpu-monitor.git
cd pytest-gpu-monitor

# Install in development mode
pip install -v -e .
```

## 🎯 Quick Start

### Basic Usage

```bash
# Enable GPU monitoring
pytest --gpu-monitor

# Run with verbose output
pytest --gpu-monitor -v -s

# Specify custom report directory
pytest --gpu-monitor --gpu-report-dir=my_reports

# Disable console summary (only generate files)
pytest --gpu-monitor --gpu-no-summary
```

### Example Output

```bash
$ pytest --gpu-monitor -v

============================================================
GPU Device: NVIDIA GeForce RTX 3090
Total GPU Memory: 24.00 GB
============================================================

test_model.py::test_small_model PASSED
[GPU] test_model.py::test_small_model
  Peak Allocated: 128.45 MB | Duration: 0.123s

test_model.py::test_large_model PASSED
[GPU] test_model.py::test_large_model
  Peak Allocated: 2048.67 MB | Duration: 1.234s

============================================================
GPU Memory Report Generated:
  JSON: gpu_memory_reports/gpu_memory_report.json
  Markdown: gpu_memory_reports/gpu_memory_report.md
  CSV: gpu_memory_reports/gpu_memory_report.csv
  HTML: gpu_memory_reports/gpu_memory_report.html
============================================================

Top 5 GPU Memory Consumers:
  1. test_model.py::test_large_model
     Peak: 2048.67 MB | Duration: 1.234s
  2. test_inference.py::test_batch_processing
     Peak: 1024.32 MB | Duration: 0.567s
  ...
```

## 📊 Report Features

### HTML Report (Recommended)

The HTML report provides an interactive dashboard with:

- **Sortable columns** - Click any column header to sort
- **Search functionality** - Filter tests by name
- **Quick filters** - View tests by memory consumption level
  - High Memory (>1GB)
  - Medium Memory (500MB-1GB)
  - Low Memory (<500MB)
- **Color coding** - Visual highlighting of high-memory tests
- **Summary statistics** - Overview cards with key metrics

### Other Formats

- **JSON** - Structured data for programmatic analysis
- **Markdown** - Human-readable text format
- **CSV** - Import into Excel or data analysis tools

## 🔧 Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu-monitor` | Enable GPU memory monitoring | `False` |
| `--gpu-report-dir` | Directory for saving reports | `gpu_memory_reports` |
| `--gpu-no-summary` | Disable console output | `False` |

## 💡 Usage Examples

### Example 1: Basic Test Suite

```bash
# tests/test_models.py
import torch
import torch.nn as nn

def test_small_model():
    model = nn.Linear(100, 100).cuda()
    x = torch.randn(32, 100).cuda()
    output = model(x)
    assert output.shape == (32, 100)

def test_large_model():
    model = nn.Linear(5000, 5000).cuda()
    x = torch.randn(128, 5000).cuda()
    output = model(x)
    assert output.shape == (128, 5000)
```

```bash
# Run tests with GPU monitoring
pytest tests/test_models.py --gpu-monitor -v
```

### Example 2: CI/CD Integration

```yaml
# .github/workflows/test.yml
name: GPU Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install pytest torch pytest-gpu-monitor
      
      - name: Run tests with GPU monitoring
        run: |
          pytest --gpu-monitor --gpu-no-summary
      
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: gpu-reports
          path: gpu_memory_reports/
```

### Example 3: Custom Report Location

```bash
# Save reports to a specific directory
pytest --gpu-monitor --gpu-report-dir=results/gpu_analysis

# The reports will be saved in:
# results/gpu_analysis/gpu_memory_report.html
# results/gpu_analysis/gpu_memory_report.json
# results/gpu_analysis/gpu_memory_report.md
# results/gpu_analysis/gpu_memory_report.csv
```

## 🔍 Metrics Explained

| Metric | Description |
|--------|-------------|
| **Peak Allocated** | Maximum GPU memory allocated during test |
| **Duration** | Test execution time in seconds |
| **Memory Increase** | Difference between final and initial memory |
| **Peak Reserved** | Maximum memory reserved by CUDA |
| **Initial Allocated** | GPU memory before test started |
| **Final Allocated** | GPU memory after test completed |

## ⚙️ How It Works

1. **Before each test**: Clears GPU cache and resets memory statistics
2. **During test**: Records memory usage at key points
3. **After test**: Calculates peak usage and stores metrics
4. **After all tests**: Generates comprehensive reports in multiple formats

## 🐛 Troubleshooting

### Plugin not found

```bash
# Verify installation
pip list | grep pytest-gpu-monitor

# Reinstall if needed
pip install --force-reinstall pytest-gpu-monitor
```

### CUDA not available

The plugin will automatically skip monitoring if CUDA is not available. Ensure:
- PyTorch is installed with CUDA support
- NVIDIA drivers are properly installed
- GPU is accessible

### Reports not generated

Check that:
- At least one test was executed
- CUDA is available (`torch.cuda.is_available()`)
- You have write permissions in the report directory

## 🗺️ Roadmap

- [ ] Support for multi-GPU setups
- [ ] Memory leak detection
- [ ] Historical trend analysis
- [ ] Integration with pytest-html
- [ ] Real-time monitoring dashboard
- [ ] Configurable thresholds and alerts
