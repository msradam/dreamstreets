# ◉ DreamStreets: AI-Powered Street Network Analysis with GPT-OSS-120b

Author: Adam Munawar Rahman, September 2025

**OpenAI Open Model Hackathon Submission** - Extending [AskStreets](https://devpost.com/software/askstreets-querying-and-visualizing-street-networks) with GPT-OSS-120b's Advanced Reasoning

DreamStreets uniquely combines OpenAI's most powerful open-source model (120 billion parameters) with established GPU-accelerated scientific computing infrastructure. By integrating GPT-OSS-120b with RAPIDS AI notebooks, we demonstrate how LLMs can augment traditional data science workflows for urban planning and humanitarian applications.


## ► What It Does

DreamStreets leverages GPT-OSS-120b alongside RAPIDS' GPU-accelerated data science stack to:
- Generate and execute NetworkX graph algorithms for network analysis
- Create spatial SQL queries for POI and facility analysis  
- Perform chain-of-thought reasoning to break down complex urban planning questions
- Provide actionable insights for both urban environments and humanitarian contexts

### Key Applications
- **Urban Planning**: Optimal business locations, infrastructure vulnerability assessment
- **Public Health**: Healthcare desert identification, emergency route planning
- **Humanitarian Response**: Refugee camp resource placement, flood preparedness
- **Emergency Management**: Evacuation planning, critical bottleneck identification

## ⚙ Technical Architecture

### AI & Scientific Computing Stack
- **LLM**: GPT-OSS-120b (120B parameters) via Ollama - completely offline
- **Scientific Computing**: RAPIDS AI 25.10a with CUDA 12.9 for GPU-accelerated data processing
- **Graph Analysis**: NetworkX Python library with built-in graph algorithm methods
- **Spatial Database**: DuckDB with spatial extensions
- **Agent Framework**: LangGraph ReAct agent with specialized tools
- **Data Source**: OpenStreetMap via OSMnx

### Why RAPIDS + LLMs?
This project demonstrates the powerful synergy between established scientific computing tools and cutting-edge language models. The RAPIDS ecosystem provides battle-tested GPU acceleration for data manipulation, while GPT-OSS-120b adds natural language understanding and code generation capabilities. This combination enables domain experts to leverage complex algorithms without writing code.

## ■ Deployment on Runpod

### Container Configuration
- **Base Image**: `rapidsai/notebooks:25.10a-cuda12.9-py3.12-amd64`
- **Environment**: JupyterLab with RAPIDS GPU-accelerated libraries

### Hardware Requirements
- **GPU**: NVIDIA A100 SXM (80GB VRAM)
- **System RAM**: 250GB
- **Storage**: 100GB container disk + 50GB persistent volume at `/home/rapids/workspace`
- **Cost**: $1.74/hr (or $1.55/hr discounted)

### Setup Instructions

1. Deploy the dreamstreets template on Runpod with exposed ports:
   - 8888 (JupyterLab)
   - 11434 (Ollama API)
   - 22 (SSH)

2. Install Ollama and the GPT-OSS-120b model:
```bash
# Navigate to workspace
cd /home/rapids/workspace

# Download and install Ollama
curl -LO https://ollama.com/download/ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz
chmod +x bin/ollama

# Start Ollama server with custom models directory
OLLAMA_MODELS=/home/rapids/workspace/.ollama/models bin/ollama serve &

# Wait for server to initialize
sleep 5

# Pull the GPT-OSS-120b model (this will take time - ~70GB download)
bin/ollama pull gpt-oss:120b
```

3. Clone the repository and install dependencies:
```bash
# Clone the dreamstreets repository
git clone https://github.com/msradam/dreamstreets.git
cd dreamstreets

# Install Python requirements
pip install -r requirements.txt
```

4. Launch Jupyter and open the notebook:
```bash
jupyter lab --ip=0.0.0.0 --port=8888
```

Then navigate to the provided URL and open `dreamstreets.ipynb` to start analyzing street networks!

## ◐ Usage Examples

### Urban Planning Query
```python
result = analyze(
    "Which intersection has the highest foot traffic based on betweenness centrality? 
     Show me the top 5 locations for a new coffee shop."
)
```

### Humanitarian Response Query
```python
result = analyze(
    "Find articulation points that would isolate communities if flooded. 
     These need elevated platforms for supply distribution during monsoons."
)
```

## ★ Key Achievements

- **Chain-of-thought reasoning** for complex spatial problems
- **Dynamic code generation** for NetworkX algorithms and SQL queries
- **Multi-tool orchestration** combining graph and database analysis
- **Complete offline operation** - critical for field deployment
- **Real-world impact** on urban planning and humanitarian response

## ═ Performance

- Analyzes networks with thousands of nodes in seconds
- Single model initialization for entire session (30-second warmup)
- Processes complex multi-step queries with up to 25 reasoning iterations
- No ongoing API costs - fully local after setup

## ● Impact Examples

### Chinatown, NYC
- Identified optimal coffee shop locations using betweenness centrality
- Found critical bottlenecks for emergency response planning
- Located healthcare deserts for mobile clinic placement

### Cox's Bazar Refugee Camp
- Optimized evacuation center placement for 1M+ residents
- Identified flood-vulnerable articulation points
- Mapped emergency medical routes requiring night lighting