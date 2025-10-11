# Introduction

"An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks."

## What an agent comprises of?

- The brain (AI model)
  - This is where all the thinking happens. The AI model handles reasoning and planning. It decides which Actions to take based on the situation.
- The body (Capabilities and Tools)
  - This part represents everything the Agent is equipped to do.

# Messages: The Underlying System of LLMs
## System Messages

System messages (also called System Prompts) define how the model should behave. They serve as persistent instructions, guiding every subsequent interaction.

```
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```
System Message also gives,
- information about the available tools
- provides instructions to the model on how to format the actions to take
- includes guidelines on how the thought process should be segmented.

## Conversations: User and Assistant Messages
A conversation consists of alternating messages between a Human (user) and an LLM (assistant).
Chat templates help maintain context by preserving conversation history, storing previous exchanges between the user and the assistant. 

```
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

We always concatenate all the messages in the conversation and pass it to the LLM as a single stand-alone sequence. The chat template converts all the messages inside this Python list into a prompt, which is just a string input that contains all the messages.

## Chat-Templates
Chat templates are essential for structuring conversations between language models and users. They guide how message exchanges are formatted into a single prompt.

# Tools
It's a function given to the LLM. Few examples
- web search --> allows agent to fetch information from the web.
- image generation --> creates images based on text descriptions.
- retrieval --> retrieves information from an external source.
- API interface --> interacts with an external API.

A tool should contain,
1. textual description of what the function does.
2. a callable (something to perform an action).
3. Arguments with typings.

# Model Context Protocol (MCP): a unified tool interface
Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools to LLMs.

MCP provides:
1. A growing list of pre-built integrations that your LLM can directly plug into
2. The flexibility to switch between LLM providers and vendors
3. Best practices for securing your data within your infrastructure