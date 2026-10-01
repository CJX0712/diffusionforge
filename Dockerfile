FROM python:3.13-slim

WORKDIR /app

# 安装运行 + 测试关键依赖（torch/diffusers 为可选 SOTA 后端）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir numpy scipy scikit-learn pytest

COPY . .

# 默认跑端到端基准，落盘 benchmark.json
CMD ["python", "diffusionforge/examples/run_demo.py"]
