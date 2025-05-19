## Agent

Agents are programs where LLM outputs control the workflow, they are useful when you need an LLM to determine the workflow of an app.

## 1. requirements

- 编译型 (Dify) 固定工作流 or 解释型 (Manus) 自主规划决策
- Router: LLM output determines an if/else switch
- Tool Caller: LLM output determines function execution
- Multistep Agent: LLM output controls iteration and program continuation
- Multi-agent: One agentic workflow can start another agentic workflow
- 记忆：长期记忆 or 短期记忆

## 2. ML task & Pipeline

- LLM (具备function call 能力)
- Tools: Plugins, Function Call, Code Interpreter
- Planning: CoT, ToT, ReAct
- Memory
- Self-Reflection / Self-Correction


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


## 7. monitor & maintain


## Reference
- https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- https://blog.langchain.dev/how-to-think-about-agent-frameworks/
- https://www.anthropic.com/engineering/building-effective-agents
- [developers.generativeai.google](https://developers.generativeai.google/develop/sample-apps/wordcraft)
- [generative-ai-docs](https://github.com/google/generative-ai-docs/tree/main/demos/palm/python/docs-agent)
- [Infrastructure for a RAG-capable generative AI application](https://cloud.google.com/architecture/rag-capable-gen-ai-app-using-vertex-ai)
