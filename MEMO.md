1. Neo4j 
   - CREATE OR REPLACE DATABASE neo4j
   - MATCH (n) RETURN (n)
   - Open Neo4j Desktop,Click on "Manage" then "Settings",Look for the "Database location" setting ,Change the path to your desired location
2. Gradio
   - filepath in gradio: gradio_api/file={file}
   - dynamic dropdown must update choices before value
3. ChromaDB
   - Chroma needs to install Visual Studio Build Tools - Microsoft
    https://visualstudio.microsoft.com/fr/downloads/?q=build+tools
   - ChromeDB collection name must be english
4. HuggingFace
   - dotenv must come first, else app blocks
   - HF_HUB_OFFLINE=1 for offline use
5. Qianfan
   - 0 < Qianfan Temperature <=1
6. langchain
    - os.environ["LANGCHAIN_TRACING_V2"] = "false"
7. Windows
   - https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
8. uv
   - windows: setx UV_DEFAULT_INDEX "https://mirrors.aliyun.com/pypi/simple"
   - uv init explore-uv
   - uv add scikit-learn xgboost : UV also updates the pyproject.toml and uv.lock files after each add command. 
   - uv remove scikit-learn
   - uv run hello.py : The run command ensures that the script is executed inside the virtual environment UV created for the project. 
   - uv python list --only-installed
   - uv python list : all available versions
   - uv tool run black hello.py or uvx black hello.py
   - uv export -o requirements.txt ： for production
   - uv add requests=2.1.2 or uv add 'requests<3.0.0'
   - uv add 'requests; sys_platform="linux"' 
   - uv add pandas or uv add pandas --optional plot excel
   - uv sync --upgrade
   - pytorch https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies
9. git
   - git rm --cached -r filename,  make a file ignored when already in .gitignore
10. vscode
    - configure PYTHONPATH=. in .env
11. Pycharm
    - configure with interpreter settings or Settings > Tools > Terminal You can add startup commands to set PYTHONPATH
    - 左上角选择Project Files，而不是Project，.idea文件就会出来
12. Graph Transformer 
    - Graph Document contains document source
13. Python
    - Thread is not interruptable by Ctrl-C, used for atomic function
14. smolagents in streaming token None, token.content None, 
15. clash, 为了避免 http://127.0.0.1:7860 不能启动的情况，在"C:\Users\PHILIPS\.config\clash-verge\clash-verge.yaml"中添加
rules:
  - DOMAIN-SUFFIX,localhost,DIRECT
  - DOMAIN,127.0.0.1,DIRECT
  - IP-CIDR,127.0.0.0/8,DIRECT  # 放行整个 127.x.x.x 环回地址
  # 其他规则...
  - GEOIP,CN,DIRECT             # 国内 IP 直连
  - MATCH,Proxy                 # 默认走代理
1. MCP
   - 百度MCP https://github.com/baidu-maps/mcp  获取sk时，要勾选mcp sse
   - StructuredTool does not support sync invocation.
   - AgentExecutor does nou support MCP client
   -   "gaode": {
    "url": "https://mcp.amap.com/sse?key=c6ff6da0179d80f11f3aea774ac1b7ff",
    "transport": "sse"
  },
   -   "baidu-maps": {
    "url": "https://mcp.map.baidu.com/sse?ak=HGSviofLzPPGM1Pz9qCPvq6a1WPAeli6",
    "transport": "sse"
  }
   - blade 密码  GGJ959zBm6szDmCT
