# Agent

Agents are programs where LLM outputs control the workflow, they are useful when you need an LLM to determine the workflow of an app.

## 1. requirements

功能性
- 编译型 (Dify) 固定工作流 or 解释型 (Manus) 自主规划决策

非功能
- latency
- throughput

## 2. ML task & Pipeline

Agent 选择
- Predefined Agents	
- Dynamically Orchestrated Agents

Agent component
- LLM (具备function call, long context 能力)
- Router: LLM output determines an if/else switch
- Tools: Plugins, Function Call, Code Interpreter
- Planning: CoT, ToT, ReAct
- Memory: 长期记忆 or 短期记忆
- Self-Reflection / Self-Correction
- Multistep Agent: LLM output controls iteration and program continuation
- Multi-agent: One agentic workflow can start another agentic workflow

Service:
- LLM Chat service (ray + VLLM)
- Agent service
  - 会话管理和上下文管理
- Tool service


## 3. Data




## 4. Model
```text
<|im_start|>system
In this environment you have access to a set of tools you can use to assist with the user query. You may perform multiple rounds of function calls. In each round, you can call one or more functions.  Here are available functions in JSONSchema format:  
\```json 
tool_schema 
\```  

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. The results of the function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's question. Finally, if you have got the answer, enclose it within \boxed{} with latex format and do not continue to call functions, i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \[ \boxed{20C} \].  

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags: 
<tool_call> 
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
{user question}
<|im_end|>
<|im_start|>assistant
<think> {think process} </think>
<tool_call> 
{"name": "get_user_checked_out_books", "arguments": {"user_id": 1}}
</tool_call>
<tool_call>
{"name": "search_books_by_author", "arguments": {"author": "Jane Doe"}} 
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response> {"name": "get_user_checked_out_books", "arguments": {"user_id": 1}}
['Python Basics', 'Advanced Python', 'Data Structures'] 
</tool_response>
<tool_response> {"name": "search_books_by_author", "arguments": {"author": "Jane Doe"}}
[{'title': 'Python Basics', 'author': 'Jane Doe', 'copies_available': 3}, {'title': 'Advanced Python', 'author': 'Jane Doe', 'copies_available': 0}] 
</tool_response>
<|im_end|>
<|im_start|>assistant
<think> ... ...
```


## 5. evaluation


## 6. deploy & service
推理加速
- KV cache
- speculative decoding
- Flash attention
- 模型量化
- Manus 的 Append-only (KV Cache) 策略
- Cursor 的实践表明，不一次性加载所有工具，而是根据当前任务只提供相关工具，能显著减少初始化Prompt的 Token 消耗。让 Agent 先用 tail 命令预览，需要时再完整读取，从而避免了因粗暴截断而丢失重要信息的问题。当任务可以分解时，可以引入 sub-agent来实现上下文隔离。


## 7. monitor & maintain


## QA
- 如何训练function call?
  - 数据集SFT 或 强化学习

- 如何解决幻觉?

- 如何解决工具调用，调用失败或返回异常数据？

- 如何缓解上下文增长？
  - 压缩 COT：长/短/空 COT 混合训练，或引入类似 KIMI 的过度思考惩罚策略
  - 滑窗机制：仅保留最近几轮的 Obs & Results。
  - 记忆剪枝：摒弃单纯的 Append-only。引入 Working Memory 和 Episodic Memory，每隔 K 步生成 State Summary 替换原始对话历史。
  - 文件外挂：类似 Manus 的策略，将较长的输出 Offload 到 Notebook 或文件中，建模为工具调用，而非占用 Prompt Token（AgentMem, FoldGRPO et al.）
  - Tool Search，不再一开始就加载所有工具定义，而是按需发现工具


## Reference
- https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- https://blog.langchain.dev/how-to-think-about-agent-frameworks/
- https://www.anthropic.com/engineering/building-effective-agents
- [developers.generativeai.google](https://developers.generativeai.google/develop/sample-apps/wordcraft)
- [generative-ai-docs](https://github.com/google/generative-ai-docs/tree/main/demos/palm/python/docs-agent)
- [Infrastructure for a RAG-capable generative AI application](https://cloud.google.com/architecture/rag-capable-gen-ai-app-using-vertex-ai)
- [What We’ve Learned From A Year of Building with LLMs](https://applied-llms.org/)

**course**
- [CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)
