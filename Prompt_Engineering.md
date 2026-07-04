# Prompt Engineering

- **LLMs** are advanced AI systems trained on massive datasets to understand, summarize, and generate human-like text.
- **Prompt engineering** is the process of designing high-quality prompts that guide LLMs to produce accurate outputs.
- In the context of **natural language processing and LLMs**, a prompt is an input provided to the model to generate a response or prediction.
- **Output Length:** An important configuration setting is the number of tokens to generate in a response.
- **Sampling controls:** LLMs do not formally predict a single token. Rather, **LLMs predict probabilities for what the next token could be**, with each token in the LLM's vocabulary getting a probability.
- **Temperature**, **top-K**, and **top-P** are the most common configuration settings.
- **Temperature** controls the degree of randomness in token selection — how "safe" vs. "adventurous" the model's word choices are. It does this by reshaping the probability distribution before sampling.
- A temperature of 0 (**greedy decoding**) is deterministic. As temperature gets higher and higher, all tokens become equally likely to be the next predicted token.
- **Greedy decoding** means the model always chooses the single most likely next token at each step.
- **Top-K and top-P:** Like temperature, these sampling settings control the randomness and diversity of generated text.
- **Top-K** sampling selects the top K most likely tokens from the model's predicted distribution. A top-K of 1 is equivalent to greedy decoding.
- **Top-P** sampling selects the top tokens whose cumulative probability does not exceed a certain value (P). Values for P range from 0 (greedy decoding) to 1 (all tokens in the LLM's vocabulary).
- Temperature, top-K, and top-P control how an AI chooses its next word: temperature controls how random or creative the output is, top-K limits the AI to the top *K* most likely words, and top-P limits it to the smallest group of words whose combined probability reaches *P*. If one setting is pushed to an extreme (like temperature = 0 or top-K = 1), it overrides the others — a good starting point for balanced results is **temperature = 0.2, top-P = 0.95, top-K = 30**, while temperature = 0 is best for tasks with a single correct answer.

## Prompting Techniques

### 1. General Prompting / Zero-Shot

It only provides a description of a task and some text for the LLM to get started with. The name zero-shot stands for "no examples."

### 2. One-Shot & Few-Shot

- **One-shot prompt:** Provides a single example, hence the name one-shot. The idea is the model has an example it can imitate to best complete the task.
- **Few-shot prompt:** Provides multiple examples to the model. This approach shows the model a pattern that it needs to follow. As a general rule of thumb, use at least three to five examples for few-shot prompting.

> When you choose examples for your prompt, use examples that are relevant to the task you want to perform.

### 3. System Prompting

Sets the overall context and purpose for the language model. It defines the "big picture" of what the model should be doing, like translating a language, classifying a review, etc.

A **system prompt** defines the AI's role and behavior before it receives the user's request. It helps ensure the AI follows consistent rules, such as giving only a sentiment label without any extra explanation.

**Example:**

```
System Prompt:
You are a helpful movie review classifier.
Your job is to classify movie reviews as POSITIVE, NEUTRAL, or NEGATIVE.
Respond with only one of these three labels and no explanation.

User Prompt:
Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
POSITIVE
```

### 4. Contextual Prompting

Provides specific details or background information relevant to the current conversation or task. It helps the model understand the nuances of what's being asked and tailor the response accordingly.

**Example:**

```
Context:
You are analyzing movie reviews for a streaming platform.
The platform defines:
- POSITIVE: The reviewer recommends or praises the movie.
- NEUTRAL: The reviewer is mixed or expresses no clear opinion.
- NEGATIVE: The reviewer criticizes or dislikes the movie.

Task:
Classify the following review.

Review:
"Her is a disturbing study revealing the direction humanity is headed
if AI is allowed to keep evolving, unchecked. I wish there were more
movies like this masterpiece."

Sentiment: POSITIVE
```

### 5. Role Prompting

Assigns a specific character or identity for the language model to adopt. This helps the model generate responses that are consistent with the assigned role and its associated knowledge and behavior.

**Example:**

```
Prompt:
You are an experienced movie critic.
Analyze the following review and classify its sentiment as POSITIVE,
NEUTRAL or NEGATIVE. Then briefly explain your reasoning.

Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
Sentiment: POSITIVE
Reason: Although the review describes the movie as disturbing, the
reviewer clearly praises it by calling it a "masterpiece" and
expressing a desire for more movies like it.
```

### 6. Step-Back Prompting

A technique where the AI first considers the **general concepts or principles** behind a problem before answering the specific question. This often leads to more accurate and well-reasoned responses.

**Example:**

```
Prompt:
Step 1: Before answering, think about the general principles used to
determine the sentiment of a movie review.
Step 2: Now classify the following review as POSITIVE, NEUTRAL, or
NEGATIVE.

Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
General principle: A review's sentiment is determined by the
reviewer's overall opinion, not just by the emotional words used to
describe the movie's subject.
Sentiment: POSITIVE
```

### 7. Chain of Thought (CoT)

Chain-of-thought prompting encourages the model to work through a problem step by step before giving the final answer. It is especially useful for reasoning tasks such as math, logic, or multi-step decision-making.

**Example:**

```
Prompt:
Classify the sentiment of the following movie review.
Think through the reasoning step by step before giving the final answer.

Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
Reasoning:
1. The review describes the movie as "disturbing," but this refers to
   the movie's theme, not the reviewer's opinion.
2. The reviewer calls it a "masterpiece."
3. The reviewer says they wish there were more movies like it.
4. These statements express strong praise.

Final Answer: POSITIVE
```

### 8. Self-Consistency

Asks the AI to solve the same problem in multiple independent ways and then select the answer that appears most consistently. This technique improves accuracy on complex reasoning tasks by reducing mistakes from a single line of reasoning.

**Example:**

```
Prompt:
Classify the sentiment of the following movie review.
Generate three independent reasoning paths, then choose the sentiment
that appears most often.

Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
Reasoning Path 1 → POSITIVE
Reasoning Path 2 → POSITIVE
Reasoning Path 3 → POSITIVE

Final Answer: POSITIVE
```

### 9. Tree of Thoughts (ToT)

Lets the AI explore multiple possible reasoning paths instead of following just one. It compares these alternatives and selects the best answer.

**Example:**

```
Prompt:
Classify the sentiment of the following movie review.
Consider multiple possible interpretations before deciding on the
final sentiment.

Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
Branch 1: The word "disturbing" suggests a NEGATIVE review.
Branch 2: The phrases "masterpiece" and "I wish there were more movies
like this" show strong praise, indicating a POSITIVE review.

Evaluation: Branch 2 better reflects the reviewer's overall opinion
because the praise outweighs the description of the movie's theme.

Final Answer: POSITIVE
```

### 10. ReAct (Reason & Act)

A prompting technique where the AI alternates between reasoning and taking actions (such as searching, using a tool, or retrieving information) before arriving at a final answer. This is especially useful when the AI needs external information or must perform multiple steps to solve a problem.

**Example:**

```
Prompt:
You are an AI assistant.
Task: Determine the sentiment of the following movie review. If
needed, first identify the reviewer's opinion, then classify the
sentiment.

Review: "Her is a disturbing study revealing the direction humanity is
headed if AI is allowed to keep evolving, unchecked. I wish there were
more movies like this masterpiece."

Output:
Reason: The word "disturbing" describes the movie's theme, not the
reviewer's opinion.
Action: Identify opinion words: "masterpiece" and "I wish there were
more movies like this."
Reason: These phrases express strong praise.

Final Answer: POSITIVE
```

## Automatic Prompt Engineering (APE)

A technique where instead of manually writing prompts, you use an AI system to generate, test, and improve prompts automatically to get the best performance on a task.

**Example:**

- Prompt A: Classify the sentiment of this movie review.
- Prompt B: Determine whether the review is positive, neutral or negative.
- Prompt C: You are a sentiment analyzer. Label the review sentiment.

Test them on sample data:

- Run each prompt on a dataset of reviews.
- Measure which one gives the highest accuracy.

**Best Prompt:** "You are a sentiment analysis system. Read the review and respond with only POSITIVE, NEUTRAL, or NEGATIVE."

## Code Prompting

Code prompting is when you ask the AI to solve a problem by writing or working with code instead of just giving a text explanation. It's commonly used for programming tasks, automation, and data processing.

**Prompts for writing code:** Should clearly describe what the program should do, what inputs it takes, and what output is expected. The more specific the instructions are, the more correct and usable the generated code will be.

**Prompts for explaining code:** Ask the AI to describe how a program works in clear, easy-to-understand language. Useful for learning new code, understanding algorithms, and debugging programs.

**Prompts for translating code:** Ask the AI to convert code from one programming language to another without changing its behavior. Helps developers reuse existing logic while adapting it to a different programming language or platform.

**Prompts for debugging and reviewing code:** Ask the AI to identify errors, explain why they occur, and suggest fixes or improvements. Useful for improving code quality, readability, efficiency, and adherence to best practices.

## Multimodal Prompting

A technique where the AI uses **more than one type of input**, such as text, images, audio, or video, to understand a task and generate a response. Instead of relying only on text, it combines information from multiple sources.
