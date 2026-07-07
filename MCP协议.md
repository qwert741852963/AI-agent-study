理解了概念之后，下一步就是动手实践。MCP的“用”和“做”可以分为两个视角：**作为使用者**去配置和调用，以及**作为开发者**去构建和扩展。

### 🧑‍💻 作为使用者：在客户端中配置MCP

这是体验MCP最直接的方式。你只需要在支持MCP的客户端（如Cursor、Claude Desktop、Cherry Studio等）中进行简单配置，即可让AI连接外部工具。

以在 **Cursor** 编辑器中配置一个MCP服务器为例，流程如下：

1.  **打开设置**：在Cursor中，导航到 `文件` > `首选项` > `Cursor设置`。
2.  **找到MCP配置**：从左侧菜单中选择“**工具和集成**”，然后在“**MCP工具**”部分，点击“**新建MCP服务器**”。这会打开一个`mcp.json`配置文件。
3.  **填入服务器信息**：在`mcp.json`文件中，你需要按照规定的格式添加服务器配置。例如，要添加一个Azure的MCP服务器，配置如下：
    ```json
    {
      "mcpServers": {
        "Azure MCP Server": {
          "command": "npx",
          "args": [
            "-y",
            "@azure/mcp@latest",
            "server",
            "start"
          ]
        }
      }
    }
    ```
    *   **`command`**: 启动服务器的命令（如`npx`, `python`）。
    *   **`args`**: 传给命令的参数。
4.  **开始使用**：保存文件后，MCP服务器通常会自动启动。之后，你就可以在Cursor的AI聊天界面中，用自然语言提问了。

> **注意**：不同客户端（如VS Code、Cline）的配置入口和格式可能略有不同，但核心都是将类似的服务器信息告诉客户端。

### 👨‍🔧 作为开发者：30分钟构建你的第一个MCP服务器

如果你想为AI创造新的能力，可以自己开发MCP服务器。这个过程比想象中要简单，核心就是用代码**定义一个AI可以理解和调用的函数**。

下面我们以Python为例，创建一个提供天气查询功能的MCP服务器。

**1. 准备环境**
首先，确保你的电脑上安装了Python（3.8+）和`pip`。然后，在终端中安装MCP的Python SDK：
```bash
pip install "mcp[cli]"
```

**2. 编写服务器代码**
创建一个Python文件（例如 `weather_server.py`），写入以下代码：
```python
from mcp.server.fastmcp import FastMCP

# 1. 创建一个MCP服务器实例
mcp = FastMCP("天气助手")

# 2. 用 @mcp.tool() 装饰器定义一个工具
@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    # 这里可以替换成真实的API调用
    return f"{city} 的天气是晴朗的，气温25°C。"

# 3. 启动服务器
if __name__ == "__main__":
    mcp.run()
```
**代码解释**：
*   `FastMCP("天气助手")`：创建了一个名为“天气助手”的MCP服务器。
*   `@mcp.tool()`：这个装饰器是关键，它告诉MCP框架：下面的函数是一个可以被AI调用的“工具”。
*   `def get_weather(city: str) -> str:`：这是工具的具体实现。函数的**名称**、**参数**（`city: str`）和**文档字符串**（`"""获取指定城市的天气"""`）都会被MCP自动解析，并描述给AI，帮助AI理解何时以及如何调用它。

**3. 运行与测试**
在终端中运行这个脚本：
```bash
python weather_server.py
```
此时，一个本地的MCP服务器就运行起来了。

### 🔗 如何配合：从代码到AI的完整链路

要让你的MCP服务器真正被AI使用，还需要将它“接入”到AI客户端中。整个过程如下：

1.  **服务器启动与注册**：当你运行`weather_server.py`时，MCP服务器启动，并开始监听客户端的连接请求。
2.  **客户端连接与发现**：在AI客户端（如Cursor）中配置好该服务器后，客户端会连接到它，并请求其提供的所有“工具”清单（`tools/list`）。客户端会获得类似这样的信息：
    *   工具名称: `get_weather`
    *   描述: `获取指定城市的天气`
    *   参数: `city` (字符串类型)
3.  **用户提问与AI决策**：当你在客户端的聊天框中提问“**上海今天天气怎么样？**”时，客户端会将这个问题连同它刚刚获取的`get_weather`工具信息一起，发送给底层的大语言模型（LLM）。
4.  **模型调用与执行**：LLM理解你的问题后，会判断需要调用`get_weather`工具，并生成一个调用指令，例如：`调用工具 get_weather，参数 city="上海"`。客户端收到这个指令后，就会向你的MCP服务器发起一个“工具调用”请求。
5.  **结果返回**：你的MCP服务器执行`get_weather("上海")`函数，得到结果“上海的天气是晴朗的，气温25°C。”，并将结果返回给客户端。
6.  **最终回复**：客户端将结果交给LLM，LLM将其组织成自然的语言，最终回复你：“上海今天的天气是晴朗的，气温25°C。”

至此，就完成了一次完整的MCP交互。


## mcp.json 文件配置
`mcp.json` 是配置 MCP 服务器的核心文件，虽然不同工具的具体实现略有差异，但核心结构和参数是通用的。

### 🗂️ 配置文件位置与优先级

`mcp.json` 可以放在不同位置，实现不同范围的配置。通常，**项目级配置的优先级最高**，会覆盖全局配置。

| 配置层级 | 常见路径 (示例) | 优先级 | 说明 |
| :--- | :--- | :--- | :--- |
| **项目级 (Project)** | `<项目根目录>/.cursor/mcp.json`<br>`<项目根目录>/.vscode/mcp.json` | **最高** | 配置仅对当前项目生效，适合团队共享。 |
| **用户级 (User)** | `~/.cursor/mcp.json`<br>`~/.claude/mcp.json` | 中 | 配置对当前用户的所有项目生效。 |
| **全局 (Global)** | `~/.mcp.json` | 最低 | 作为所有用户的兜底配置。 |

> **注意**：具体的路径和文件名（如 `.cursor/` 或 `.vscode/`）会因你使用的 AI 工具（如 Cursor、VS Code、Claude Code 等）而异。

### 📝 核心配置结构

`mcp.json` 文件是一个标准的 JSON 文件，其最外层必须包含一个名为 `mcpServers` 的对象。

```json
{
  "mcpServers": {
    "your-server-name": {
      // ... 服务器配置参数
    },
    "another-server": {
      // ... 另一个服务器的配置
    }
  }
}
```

*   **`mcpServers`**: 根对象，包含了所有 MCP 服务器的配置。
*   **`"your-server-name"`**: 你为服务器定义的**唯一别名**，AI 工具通过这个名字来识别和调用不同的服务。命名通常只支持字母、数字、下划线、中划线和方括号。

### ⚙️ 服务器配置参数详解

每个服务器配置（即 `"your-server-name": { ... }` 中的内容）都支持以下参数。

#### 通用参数

| 参数名 | 类型 | 必填 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- | :--- |
| **`type` / `transportType`** | 字符串 | **是** | 指定传输类型，决定客户端如何与服务器通信。可选值：`"stdio"`, `"sse"`, `"streamableHttp"`。 | 无 |
| **`disabled`** | 布尔值 | 否 | 设为 `true` 可禁用该服务器，而无需删除配置。 | `false` |
| **`timeout`** | 数字 | 否 | 请求超时时间，单位为**秒**。 | `30` 秒 |
| **`env`** | 对象 | 否 | 以键值对的形式设置**环境变量**，常用于传递 API 密钥等敏感信息。 | `{}` |
| **`cwd`** | 字符串 | 否 | 指定**进程的工作目录**。如果未指定，默认为用户的家目录。 | 用户家目录 |

#### 传输类型特定参数

| 参数名 | 类型 | 必填 | 说明 | 适用传输类型 |
| :--- | :--- | :--- | :--- | :--- |
| **`command`** | 字符串 | **是** | 要执行的**可执行命令**，用于启动本地进程，如 `"npx"`, `"python"`, `"node"`。 | **仅限 `stdio`** |
| **`args`** | 字符串数组 | 否 | 传给 `command` 的**命令行参数列表**，如 `["-y", "@modelcontextprotocol/server-github"]`。 | **仅限 `stdio`** |
| **`url`** | 字符串 (URI) | **是** | 远程 MCP 服务器的 **HTTP 或 SSE 端点地址**。 | **仅限 `sse` 或 `streamableHttp`** |
| **`headers`** | 对象 | 否 | 以键值对形式设置**自定义 HTTP 请求头**，常用于身份验证。 | **仅限 `sse` 或 `streamableHttp`** |

### 💡 配置示例

#### 1. 本地 `stdio` 服务器 (最常用)
这是最典型的配置，通过启动一个本地进程来提供功能。
```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github@0.6.2"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "你的GitHub Token"
      }
    }
  }
}
```
*   **`github`**: 服务器别名。
*   **`type`**: 设为 `"stdio"`。
*   **`command`**: 使用 `npx` 命令。
*   **`args`**: 告诉 `npx` 去运行 `@modelcontextprotocol/server-github` 这个包。
*   **`env`**: 通过环境变量传入 GitHub 的个人访问令牌。

#### 2. 远程 `sse` 服务器
连接到已经部署在远程的 MCP 服务器。
```json
{
  "mcpServers": {
    "monitoring": {
      "type": "sse",
      "url": "https://mcp.monitoring.example.com/sse",
      "headers": {
        "Authorization": "Bearer 你的访问令牌"
      },
      "timeout": 60
    }
  }
}
```
*   **`monitoring`**: 服务器别名。
*   **`type`**: 设为 `"sse"`。
*   **`url`**: 远程服务器的 SSE 端点。
*   **`headers`**: 在 HTTP 请求头中携带 Bearer Token 进行认证。

### ✨ 进阶用法：变量占位符

部分工具支持在配置中使用变量占位符，让配置更灵活。例如，在 `command`、`args`、`env` 或 `cwd` 字段中，你可以使用：
*   `$COMATE.ENV.WORKSPACE`: 代表当前项目的根目录路径。
*   `$COMATE.ENV.USERNAME`: 代表当前系统的用户名。

这使得路径配置不再依赖于绝对路径，增强了配置文件在不同机器和环境下的可移植性。

### ⚠️ 常见问题与注意事项

*   **配置未生效**：检查文件路径是否正确，以及 `disabled` 字段是否被意外设为 `true`。
*   **命令找不到**：确认 `command` 对应的程序已安装，或检查 `cwd` 工作目录是否正确。
*   **认证失败**：仔细检查 `env` 或 `headers` 中的密钥、令牌等敏感信息是否正确无误。
*   **超时错误**：如果服务器响应较慢，可以尝试增大 `timeout` 的值。

如果遇到问题，可以先检查 AI 工具的日志或输出面板，通常会包含详细的错误信息来帮助你定位问题。






