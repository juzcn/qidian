import asyncio

from fastmcp import Client


async def main():
    # 测试 mcp 客户端的功能
    async with Client("http://127.0.0.1:8001/sse") as mcp_client:
        tools = await mcp_client.list_tools()
        print(f"Available tools: {tools}")
        result = await mcp_client.call_tool("add", {"a": 5, "b": 3})
        print(f"Result: {result[0].text}")

asyncio.run(main())