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
