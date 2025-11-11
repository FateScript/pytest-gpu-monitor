"""
pytest-gpu-monitor: A pytest plugin to monitor GPU memory usage during tests
"""

import json
import time
from csv import DictWriter
from datetime import datetime
from pathlib import Path

import pytest
import torch

GPU_MEMORY_DATA = []  # to store GPU memory data for all tests
NUM_TESTS = 0


def is_xdist_worker(request_or_session: pytest.FixtureRequest | pytest.Session) -> bool:
    """Return `True` if this is an xdist worker, `False` otherwise."""
    return hasattr(request_or_session.config, "workerinput")


def pytest_xdist_node_collection_finished(node, ids):
    """To count the total number of tests across all workers."""
    global NUM_TESTS
    NUM_TESTS = len(ids)


def pytest_addoption(parser):
    group = parser.getgroup("gpu-monitor")
    group.addoption(
        "--gpu-monitor",
        action="store_true",
        default=False,
        help="Enable GPU memory monitoring",
    )
    group.addoption(
        "--gpu-report-dir",
        action="store",
        default="gpu_memory_reports",
        help="Directory to save GPU memory reports (default: gpu_memory_reports)",
    )
    group.addoption(
        "--gpu-no-summary",
        action="store_true",
        default=False,
        help="Disable printing summary to console",
    )
    return parser


def pytest_configure(config: pytest.Config):
    """config before tests start"""
    if not config.getoption("--gpu-monitor"):
        return

    global GPU_MEMORY_DATA
    GPU_MEMORY_DATA = []  # reset data at the start of the session

    # log info
    # NOTE: pytest-xdist set `@pytest.hookimpl(trylast=True)` for `pytest_configure`
    if torch.cuda.is_available() and not config.getoption("--gpu-no-summary"):
        print(f"\n{'=' * 60}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")  # noqa
        if config.option.dist != "no":
            print(f"Running with pytest-xdist: {config.option.numprocesses} workers")
        print(f"{'=' * 60}\n")


@pytest.fixture(scope="session", autouse=True)
def assign_gpu_to_worker(request):
    # xdist worker ID: gw0, gw1, etc.
    if not torch.cuda.is_available():
        return
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "gw0")
    gpu_id = int(worker_id.replace("gw", "")) % torch.cuda.device_count()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)


@pytest.fixture(autouse=True)
def monitor_gpu_memory(request):
    """Monitor GPU memory usage and execution time for each test"""
    config = request.config

    # skip if not using `--gpu-monitor` flag / CUDA not available
    if not config.getoption("--gpu-monitor"):
        yield
        return

    test_name = request.node.nodeid

    if not torch.cuda.is_available():
        yield
        return

    # clean up cuda memory before test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    initial_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    initial_reserved = torch.cuda.memory_reserved() / 1024**2
    start_time = time.time()

    yield

    # synchronize and measure after test finishes
    torch.cuda.synchronize()
    end_time = time.time()

    final_allocated = torch.cuda.memory_allocated() / 1024**2
    final_reserved = torch.cuda.memory_reserved() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    worker_id = getattr(config, "workerinput", {}).get("workerid", "gw0") if is_xdist_worker(request) else "master"  # noqa

    test_data = {
        "test_name": test_name,
        "duration_seconds": round(end_time - start_time, 3),
        "initial_allocated_mb": round(initial_allocated, 2),
        "final_allocated_mb": round(final_allocated, 2),
        "peak_allocated_mb": round(peak_allocated, 2),
        "memory_increase_mb": round(final_allocated - initial_allocated, 2),
        "initial_reserved_mb": round(initial_reserved, 2),
        "final_reserved_mb": round(final_reserved, 2),
        "peak_reserved_mb": round(peak_reserved, 2),
        "timestamp": datetime.now().isoformat(),
        "worker_id": worker_id,
    }

    if is_xdist_worker(request):
        report_dir = Path(config.getoption("--gpu-report-dir"))
        report_dir.mkdir(exist_ok=True)
        temp_dir = report_dir / ".temp"
        temp_dir.mkdir(exist_ok=True)

        temp_file = temp_dir / f"gpu_data_{worker_id}_{test_name.replace('::', '_').replace('/', '_')}.json"  # noqa
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2)
    else:
        GPU_MEMORY_DATA.append(test_data)

    # print summary (optional)
    if not config.getoption("--gpu-no-summary"):
        print(f"\n[GPU] {test_name}")
        print(f"  Peak Allocated: {peak_allocated:.2f} MB | Duration: {test_data['duration_seconds']:.3f}s")  # noqa


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """After all tests have finished, generate the report"""
    # NOTE: `pytest_sessionfinish` is too early to read xdist data
    use_xdist = config.option.dist != "no"

    if not config.getoption("--gpu-monitor"):
        return

    if not torch.cuda.is_available():
        return

    report_dir = Path(config.getoption("--gpu-report-dir"))
    report_dir.mkdir(exist_ok=True)
    temp_dir = report_dir / ".temp"

    all_data = []

    if use_xdist:
        assert temp_dir.exists(), "Temp directory for xdist data not found"
        num_files = 0
        while num_files < NUM_TESTS:
            local_files = list(temp_dir.glob("gpu_data_*.json"))
            num_files += len(local_files)
            for temp_file in local_files:
                try:
                    with open(temp_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_data.append(data)
                except Exception as e:
                    print(f"Warning: Failed to read {temp_file}: {e}")

                try:  # rm file after reading
                    temp_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete {temp_file}: {e}")
    else:
        all_data = GPU_MEMORY_DATA

    if not all_data:  # no data collected
        return

    # use fixed filename, overwrite each time
    json_file = report_dir / "gpu_memory_report.json"
    md_file = report_dir / "gpu_memory_report.md"
    csv_file = report_dir / "gpu_memory_report.csv"
    html_file = report_dir / "gpu_memory_report.html"

    # save detailed data in JSON format
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_tests": len(all_data),
                    "gpu_device": torch.cuda.get_device_name(0),
                    "total_gpu_memory_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1024**3,
                        2,
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "xdist_enabled": config.option.dist != "no",
                },
                "tests": all_data,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # generate report
    generate_markdown_report(md_file, all_data)
    generate_csv_report(csv_file, all_data)
    generate_html_report(html_file, all_data)

    # top 5 memory consumers
    if not config.getoption("--gpu-no-summary"):
        print(f"{'=' * 60}\n")
        sorted_tests = sorted(all_data, key=lambda x: x["peak_allocated_mb"], reverse=True)
        print("\nTop 5 GPU Memory Consumers:")
        for i, test in enumerate(sorted_tests[:5], 1):
            print(f"  {i}. {test['test_name']}")
            print(f"     Peak: {test['peak_allocated_mb']:.2f} MB | Duration: {test['duration_seconds']:.3f}s")  # noqa


def generate_markdown_report(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# GPU Memory Usage Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if torch.cuda.is_available():
            f.write(f"**GPU Device:** {torch.cuda.get_device_name(0)}\n")
            f.write(f"**Total GPU Memory:** {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n\n")  # noqa

        f.write(f"**Total Tests:** {len(data)}\n\n")

        # statistics of each test
        total_peak = sum(t["peak_allocated_mb"] for t in data)
        avg_peak = total_peak / len(data) if data else 0
        max_peak = max(t["peak_allocated_mb"] for t in data) if data else 0

        f.write("## Summary\n\n")
        f.write(f"- Average Peak Memory: {avg_peak:.2f} MB\n")
        f.write(f"- Max Peak Memory: {max_peak:.2f} MB\n")
        f.write(f"- Total Duration: {sum(t['duration_seconds'] for t in data):.2f}s\n\n")

        # detailed results in markdown table
        f.write("## Detailed Results\n\n")
        f.write("| Test Name | Peak Memory (MB) | Duration (s) | Memory Increase (MB) |\n")
        f.write("|-----------|------------------|--------------|----------------------|\n")

        for test in sorted(data, key=lambda x: x["peak_allocated_mb"], reverse=True):
            name = test["test_name"].split("::")[-1][:50]  # 截断长名称
            f.write(
                f"| {name} | {test['peak_allocated_mb']:.2f} | "
                f"{test['duration_seconds']:.3f} | {test['memory_increase_mb']:.2f} |\n"
            )


def generate_csv_report(filepath, data):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        if not data:
            return

        writer = DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def generate_html_report(filepath, data):
    # NOTE: this frontend code is generated by Kimi K2
    total_tests = len(data)
    avg_peak = sum(t["peak_allocated_mb"] for t in data) / total_tests if total_tests else 0
    max_peak = max(t["peak_allocated_mb"] for t in data) if data else 0
    total_duration = sum(t["duration_seconds"] for t in data)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GPU Memory Usage Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}

        h1 {{
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .timestamp {{
            color: #718096;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}

        .summary-card:hover {{
            transform: translateY(-5px);
        }}

        .summary-card h3 {{
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
            margin-bottom: 10px;
        }}

        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}

        .controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .search-box {{
            flex: 1;
            min-width: 300px;
            padding: 12px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }}

        .search-box:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .filter-buttons {{
            display: flex;
            gap: 10px;
        }}

        .filter-btn {{
            padding: 10px 20px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }}

        .filter-btn:hover {{
            background: #667eea;
            color: white;
        }}

        .filter-btn.active {{
            background: #667eea;
            color: white;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            position: relative;
            transition: background 0.3s;
        }}

        th:hover {{
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }}

        th::after {{
            content: '⇅';
            position: absolute;
            right: 15px;
            opacity: 0.5;
            font-size: 0.9em;
        }}

        th.sort-asc::after {{
            content: '↑';
            opacity: 1;
        }}

        th.sort-desc::after {{
            content: '↓';
            opacity: 1;
        }}

        td {{
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            transition: background 0.2s;
        }}

        tr:hover td {{
            background-color: #f7fafc;
        }}

        .test-name {{
            font-family: 'Courier New', monospace;
            color: #2d3748;
            font-size: 0.9em;
        }}

        .memory-value {{
            font-weight: 600;
            color: #667eea;
        }}

        .high-memory {{
            background-color: #fff5f5 !important;
        }}

        .high-memory .memory-value {{
            color: #e53e3e;
        }}

        .medium-memory {{
            background-color: #fffaf0 !important;
        }}

        .medium-memory .memory-value {{
            color: #dd6b20;
        }}

        .duration-badge {{
            display: inline-block;
            padding: 4px 12px;
            background-color: #edf2f7;
            border-radius: 12px;
            font-size: 0.9em;
            color: #4a5568;
        }}

        .stats-footer {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f7fafc;
            border-radius: 8px;
            text-align: center;
            color: #718096;
        }}

        #noResults {{
            text-align: center;
            padding: 40px;
            color: #718096;
            font-size: 1.1em;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GPU Memory Usage Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <div class="summary-card">
                <h3>GPU Device</h3>
                <div class="value" style="font-size: 1.2em;">{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}</div>
            </div>
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{total_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Peak Memory</h3>
                <div class="value">{avg_peak:.1f} <span style="font-size: 0.5em;">MB</span></div>
            </div>
            <div class="summary-card">
                <h3>Max Peak Memory</h3>
                <div class="value">{max_peak:.1f} <span style="font-size: 0.5em;">MB</span></div>
            </div>
            <div class="summary-card">
                <h3>Total Duration</h3>
                <div class="value">{total_duration:.1f} <span style="font-size: 0.5em;">s</span></div>
            </div>
        </div>

        <h2 style="margin-bottom: 15px; color: #2d3748;">📊 Detailed Results</h2>

        <div class="controls">
            <input type="text" id="searchBox" class="search-box" placeholder="🔍 Search tests...">
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterTests('all')">All</button>
                <button class="filter-btn" onclick="filterTests('high')">High Memory (&gt;1GB)</button>
                <button class="filter-btn" onclick="filterTests('medium')">Medium (&gt;500MB)</button>
                <button class="filter-btn" onclick="filterTests('low')">Low (&lt;500MB)</button>
            </div>
        </div>

        <table id="dataTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)" data-type="string">Test Name</th>
                    <th onclick="sortTable(1)" data-type="number">Peak Allocated (MB)</th>
                    <th onclick="sortTable(2)" data-type="number">Duration (s)</th>
                    <th onclick="sortTable(3)" data-type="number">Memory Increase (MB)</th>
                    <th onclick="sortTable(4)" data-type="number">Peak Reserved (MB)</th>
                    <th onclick="sortTable(5)" data-type="number">Initial Allocated (MB)</th>
                </tr>
            </thead>
            <tbody id="tableBody">
"""

    # 添加测试数据
    for test in data:
        peak_mb = test["peak_allocated_mb"]
        row_class = ""
        if peak_mb > 1000:
            row_class = "high-memory"
        elif peak_mb > 500:
            row_class = "medium-memory"

        html_content += f"""
                <tr class="{row_class}">
                    <td class="test-name">{test["test_name"]}</td>
                    <td class="memory-value">{test["peak_allocated_mb"]:.2f}</td>
                    <td><span class="duration-badge">{test["duration_seconds"]:.3f}</span></td>
                    <td>{test["memory_increase_mb"]:.2f}</td>
                    <td>{test["peak_reserved_mb"]:.2f}</td>
                    <td>{test["initial_allocated_mb"]:.2f}</td>
                </tr>
"""

    html_content += f"""
            </tbody>
        </table>

        <div id="noResults">
            <p>No tests match your search criteria.</p>
        </div>

        <div class="stats-footer">
            <p>Showing <span id="visibleRows">{len(data)}</span> of {len(data)} tests</p>
        </div>
    </div>

    <script>
        let sortDirection = {{}};
        let currentFilter = 'all';

        // 排序功能
        function sortTable(columnIndex) {{
            const table = document.getElementById('dataTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const header = table.querySelectorAll('th')[columnIndex];
            const dataType = header.getAttribute('data-type');

            // 切换排序方向
            const currentDirection = sortDirection[columnIndex] || 'none';
            let newDirection;

            if (currentDirection === 'none' || currentDirection === 'desc') {{
                newDirection = 'asc';
            }} else {{
                newDirection = 'desc';
            }}

            sortDirection = {{}};
            sortDirection[columnIndex] = newDirection;

            // 更新表头样式
            table.querySelectorAll('th').forEach(th => {{
                th.classList.remove('sort-asc', 'sort-desc');
            }});
            header.classList.add('sort-' + newDirection);

            // 排序
            rows.sort((a, b) => {{
                let aValue = a.cells[columnIndex].textContent.trim();
                let bValue = b.cells[columnIndex].textContent.trim();

                if (dataType === 'number') {{
                    aValue = parseFloat(aValue) || 0;
                    bValue = parseFloat(bValue) || 0;
                    return newDirection === 'asc' ? aValue - bValue : bValue - aValue;
                }} else {{
                    return newDirection === 'asc' 
                        ? aValue.localeCompare(bValue)
                        : bValue.localeCompare(aValue);
                }}
            }});

            // 重新插入排序后的行
            rows.forEach(row => tbody.appendChild(row));
        }}

        // 搜索功能
        document.getElementById('searchBox').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#tableBody tr');
            let visibleCount = 0;

            rows.forEach(row => {{
                const testName = row.cells[0].textContent.toLowerCase();
                const matches = testName.includes(searchTerm);
                const passesFilter = checkFilter(row);

                if (matches && passesFilter) {{
                    row.style.display = '';
                    visibleCount++;
                }} else {{
                    row.style.display = 'none';
                }}
            }});

            updateVisibleCount(visibleCount);
        }});

        // 过滤功能
        function filterTests(type) {{
            currentFilter = type;

            // 更新按钮状态
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');

            const rows = document.querySelectorAll('#tableBody tr');
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            let visibleCount = 0;

            rows.forEach(row => {{
                const testName = row.cells[0].textContent.toLowerCase();
                const matchesSearch = testName.includes(searchTerm);
                const passesFilter = checkFilter(row);

                if (matchesSearch && passesFilter) {{
                    row.style.display = '';
                    visibleCount++;
                }} else {{
                    row.style.display = 'none';
                }}
            }});

            updateVisibleCount(visibleCount);
        }}

        function checkFilter(row) {{
            const peakMemory = parseFloat(row.cells[1].textContent);

            switch(currentFilter) {{
                case 'high':
                    return peakMemory > 1000;
                case 'medium':
                    return peakMemory > 500 && peakMemory <= 1000;
                case 'low':
                    return peakMemory <= 500;
                default:
                    return true;
            }}
        }}

        function updateVisibleCount(count) {{
            document.getElementById('visibleRows').textContent = count;
            const noResults = document.getElementById('noResults');
            const table = document.getElementById('dataTable');

            if (count === 0) {{
                noResults.style.display = 'block';
                table.style.display = 'none';
            }} else {{
                noResults.style.display = 'none';
                table.style.display = 'table';
            }}
        }}

        // 默认按Peak Allocated降序排序
        window.addEventListener('DOMContentLoaded', function() {{
            sortTable(1);
            sortTable(1);
        }});
    </script>
</body>
</html>
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
