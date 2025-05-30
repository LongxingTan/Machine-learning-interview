# 个性化标题生成

## 1. requirements

**functional**

- 语言
- 长度

**non-functional**

- throughput
- latency

## 2. ML task & pipeline

## 3. data collection

- start from zero-shot/few-shot and prompt engineering

## 4. feature

## 5. model

**生成策略**

- Greedy
- Beam Search
- Random Sampling
- Temperature
- Top-K Sampling
- Nucleus Sampling

## 6. evaluation

## 7. deploy & serving

framework

- vLLM + Huggingface TGI
- TensorRT-LLM (especially for hardware supports float8 inference)
  - Triton Inference Server with TensorRT-LLM
- sglang
- Triton Inference Server
- lmdeploy

## 8. monitor & maintenance

## reference

- [信息流场景下的AIGC实践](https://mp.weixin.qq.com/s/AOTP6oNXhtcCUhdtcEwMTg)
- [https://github.com/microsoft/promptflow](https://github.com/microsoft/promptflow)
